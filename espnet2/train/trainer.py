from __future__ import annotations

import logging
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.nn
import torch.optim
from torch.nn.parallel import data_parallel
from torch.utils.data import DataLoader
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.schedulers.abs_scheduler import AbsBatchScheduler
from espnet2.schedulers.abs_scheduler import AbsEpochScheduler
from espnet2.schedulers.abs_scheduler import AbsValEpochScheduler
from espnet2.train.abs_e2e import AbsE2E
from espnet2.train.add_gradient_noise import add_gradient_noise
from espnet2.train.reporter import Reporter
from espnet2.train.reporter import SubReporter
from espnet2.utils.calculate_all_attentions import calculate_all_attentions
from espnet2.utils.device_funcs import to_device
from espnet2.utils.fileio import DatadirWriter


class Trainer:
    @classmethod
    def load(
        cls,
        model: AbsE2E,
        optimizer: torch.optim.Optimizer,
        reporter: Reporter,
        output_dir: Union[str, Path],
        batch_scheduler: Optional[AbsBatchScheduler],
        epoch_scheduler: Optional[AbsEpochScheduler],
        resume_epoch: Optional[Union[int, str]],
        resume_path: Optional[Union[str, Path]],
        pretrain_path: Optional[Union[str, Path]],
        pretrain_key: Optional[str],
        map_location: str,
    ) -> None:
        assert check_argument_types()
        # For resuming: Specify either resume_epoch or resume_path.
        #     - resume_epoch: Load from outdir/{}epoch/.
        #     - resume_path: Load from the specified path.
        # Find the latest epoch snapshot
        if resume_epoch == "latest":
            resume_epoch = 0
            latest = None
            for p in output_dir.glob("*epoch/timestamp"):
                try:
                    n = int(p.parent.name.replace("epoch", ""))
                except TypeError:
                    continue
                with p.open("r") as f:
                    # Read the timestamp and comparing
                    date = f.read().strip()
                    try:
                        date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        continue
                if latest is None or date > latest:
                    resume_epoch = n
                    latest = date

        # If not found any snapshots, then nothing is done
        if resume_epoch == 0:
            resume_epoch = None

        if resume_epoch is not None or resume_path is not None:
            if resume_path is None:
                resume_path = output_dir / f"{resume_epoch}epoch"
                logging.info(
                    f"--resume_epoch {resume_epoch}: "
                    f"Loading from {resume_path}"
                )
            else:
                logging.info(
                    f"--resume_path {resume_path}: "
                    f"Loading from {resume_path}"
                )

            for key, obj in [
                ("model", model),
                ("optimizer", optimizer),
                ("reporter", reporter),
                ("epoch_scheduler", epoch_scheduler),
                ("batch_scheduler", batch_scheduler),
            ]:
                _st = torch.load(
                    resume_path / f"{key}.pt", map_location=map_location
                )
                if obj is not None:
                    obj.load_state_dict(_st)

        # FIXME(kamo): Should be done in Task.build_model() or in model?
        # For distillation, fine-tuning, transfer learning, etc.
        if pretrain_path is not None:
            if pretrain_key is None:
                obj = model
            else:

                def get_attr(obj: Any, key: str):
                    """

                    >>> class A(torch.nn.Module):
                    ...     def __init__(self):
                    ...         super().__init__()
                    ...         self.linear = torch.nn.Linear(10, 10)
                    >>> a = A()
                    >>> assert A.linear.weight is get_attr(A, 'linear.weight')

                    """
                    if key.strip() == "":
                        return obj
                    for k in key.split("."):
                        obj = getattr(obj, k)
                    return obj

                obj = get_attr(model, pretrain_key)

            state_dict = obj.state_dict()
            pretrained_dict = torch.load(
                pretrain_path, map_location=map_location
            )
            # Ignores the parameters not existing in the train-model
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in state_dict
            }
            state_dict.update(pretrained_dict)
            obj.load_state_dict(state_dict)

    @classmethod
    @torch.no_grad()
    def collect_stats(
        cls,
        model: AbsE2E,
        train_iter: DataLoader and Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        eval_iter: DataLoader and Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        output_dir: Union[str, Path],
        ngpu: Optional[int],
        log_interval: Optional[int],
    ) -> None:
        """Running for deriving the shape information from data
        and gathering statistics"""
        assert check_argument_types()
        output_dir = Path(output_dir)

        for itr, mode in zip([train_iter, eval_iter], ["train", "eval"]):
            if log_interval is None:
                log_interval = max(len(itr) // 20, 10)

            sum_dict = defaultdict(lambda: 0)
            sq_dict = defaultdict(lambda: 0)
            count_dict = defaultdict(lambda: 0)

            with DatadirWriter(output_dir / mode) as writer:
                for iiter, (keys, batch) in enumerate(itr, 1):
                    batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")

                    # 1. Write shape file
                    for name in batch:
                        if name.endswith("_lengths"):
                            continue
                        for i, (k, data) in enumerate(zip(keys, batch[name])):
                            if f"{name}_lengths" in batch:
                                lg = int(batch[f"{name}_lengths"][i])
                                shape = ",".join(map(str, (lg,) + data.shape[1:]))
                            else:
                                shape = ",".join(map(str, data.shape))
                            writer[f"{name}_shape"][k] = shape

                    # 2. Extract feats and calc sum and square sum
                    data = model.collect_feats(**batch)
                    for k, v in data.items():
                        if k.endswith("_lengths"):
                            continue
                        if f"{k}_lengths" in data:
                            # value: (Batch, Length, Dim, ...)
                            # -> Summation over batchxlength
                            ind = (0, 1)
                            count = v.size(0) * v.size(1)
                        else:
                            # value: (Batch, Dim, ...)
                            # -> Summation over batch
                            ind = 0
                            count = v.size(0)
                        v = v.cpu()
                        v.masked_fill_(
                            make_pad_mask(data[f"{k}_lengths"], v, 1), 0.0
                        )
                        sum_dict[k] += v.sum(ind).cpu().numpy()
                        sq_dict[k] += (v ** 2).sum(ind).cpu().numpy()
                        count_dict[k] += count

                    if iiter % log_interval == 0:
                        logging.info(f"Niter: {iiter}")

            for key in sum_dict:
                np.savez(
                    output_dir / mode / f"{key}_stats.npz",
                    count=count_dict[key],
                    sum=sum_dict[key],
                    sum_square=sq_dict[key]
                )
            with (output_dir / mode / "shape_keys").open("w") as f:
                f.write("\n".join(
                    filter(lambda x: not x.endswith("_lengths"), batch)) + "\n"
                )
            with (output_dir / mode / "stats_keys").open("w") as f:
                f.write("\n".join(sum_dict) + "\n")

    @classmethod
    def run(
        cls,
        model: AbsE2E,
        optimizer: torch.optim.Optimizer,
        train_iter: DataLoader and Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        eval_iter: DataLoader and Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        plot_attention_iter,
        reporter: Reporter,
        output_dir: Union[str, Path],
        batch_scheduler: Optional[AbsBatchScheduler],
        epoch_scheduler: Optional[AbsEpochScheduler],
        max_epoch: int,
        patience: Optional[int],
        ngpu: int,
        train_dtype: str,
        grad_noise: bool,
        accum_grad: int,
        grad_clip: float,
        log_interval: Optional[int],
        keep_n_best_snapshot: int,
        early_stopping_criterion: Sequence[str],
        best_model_criterion: Sequence[Sequence[str]],
        val_scheduler_criterion: Sequence[str],
        no_forward_run: bool,
        no_backward_run: bool,
    ) -> None:
        assert check_argument_types()

        # For apex supporting
        if train_dtype in ("O0", "O1", "O2", "O3"):
            try:
                from apex import amp
            except ImportError:
                logging.error(
                    f"You need to install apex. "
                    f"See https://github.com/NVIDIA/apex#linux"
                )
                raise
            model, optimizer = amp.initialize(
                model, optimizer, opt_level=train_dtype
            )

        start_epoch = reporter.get_epoch() + 1
        if start_epoch == max_epoch + 1:
            logging.warning(
                f"The training has already reached at "
                f"max_epoch: {start_epoch}"
            )

        best_epoch_dict = {}
        for iepoch in range(start_epoch, max_epoch + 1):
            logging.info(f"{iepoch}epoch started")

            reporter.set_epoch(iepoch)
            # 1. Train and eval for one-epoch
            with reporter.observe("train") as sub_reporter:
                all_steps_are_invalid = cls.train(
                    model=model,
                    optimizer=optimizer,
                    iterator=train_iter,
                    reporter=sub_reporter,
                    scheduler=batch_scheduler,
                    ngpu=ngpu,
                    use_apex=train_dtype in ("O0", "O1", "O2", "O3"),
                    grad_noise=grad_noise,
                    accum_grad=accum_grad,
                    grad_clip=grad_clip,
                    log_interval=log_interval,
                    no_forward_run=no_forward_run,
                    no_backward_run=no_backward_run,
                )
            with reporter.observe("eval") as sub_reporter:
                cls.eval(
                    model=model,
                    iterator=eval_iter,
                    reporter=sub_reporter,
                    ngpu=ngpu,
                    no_forward_run=no_forward_run,
                )
            if plot_attention_iter is not None and not no_forward_run:
                with reporter.observe("att_plot") as sub_reporter:
                    cls.plot_attention(
                        model=model,
                        output_dir=output_dir / "att_ws" / f"{iepoch}epoch",
                        iterator=plot_attention_iter,
                        ngpu=ngpu,
                        reporter=sub_reporter,
                    )

            # 2. Scheduler step
            #   Controls opt-params by scheduler e.g. learning rate decay
            if epoch_scheduler is not None:
                if isinstance(epoch_scheduler, AbsValEpochScheduler):
                    _phase, _criterion = val_scheduler_criterion
                    if not reporter.has(_phase, _criterion):
                        raise RuntimeError(
                            f"{_phase}.{_criterion} is not found in stats"
                            f"{reporter.get_all_keys()}"
                        )
                    val = reporter.get_value(_phase, _criterion)
                    epoch_scheduler.step(val)
                else:
                    epoch_scheduler.step()

            # 3. Report the results
            reporter.logging()
            reporter.save_stats_plot(output_dir / "images")

            # 4. Save the snapshot
            for key, obj in [
                ("model", model),
                ("optimizer", optimizer),
                ("reporter", reporter),
                ("epoch_scheduler", epoch_scheduler),
                ("batch_scheduler", batch_scheduler),
            ]:
                (output_dir / f"{iepoch}epoch").mkdir(
                    parents=True, exist_ok=True
                )
                p = output_dir / f"{iepoch}epoch" / f"{key}.pt"
                p.parent.mkdir(parents=True, exist_ok=True)
                torch.save(obj.state_dict() if obj is not None else None, p)
            # Write the datetime in "timestamp"
            with (output_dir / f"{iepoch}epoch" / "timestamp").open("w") as f:
                dt = datetime.now()
                f.write(dt.strftime("%Y-%m-%d %H:%M:%S") + "\n")

            # 5. Saves the best model
            _improved = []
            for _phase, k, _mode in best_model_criterion:
                if reporter.has(_phase, k):
                    best_epoch, _ = reporter.sort_epochs_and_values(
                        _phase, k, _mode
                    )[0]
                    best_epoch_dict[(_phase, k)] = best_epoch
                    # Creates sym links if it's the best result
                    if best_epoch == iepoch:
                        p = output_dir / f"{_phase}.{k}.best.pt"
                        if p.is_symlink() or p.exists():
                            p.unlink()
                        p.symlink_to(Path(f"{iepoch}epoch") / f"model.pt")
                        _improved.append(f"{_phase}.{k}")
            if len(_improved) == 0:
                logging.info(f"There are no improvements in this epoch")
            else:
                logging.info(
                    f"The best model has been updated: " + ", ".join(_improved)
                )

            # 6. Remove the snapshot excluding n-best and the current epoch
            _removed = []
            # nbests: List[List[Tuple[epoch, value]]]
            nbests = [
                reporter.sort_epochs_and_values(ph, k, m)[
                    :keep_n_best_snapshot
                ]
                for ph, k, m in best_model_criterion
                if reporter.has(ph, k)
            ]
            # nbests: Set[epoch]
            if len(nbests) != 0:
                nbests = set.union(*[set(i[0] for i in v) for v in nbests])
            else:
                nbests = set()
            for e in range(1, iepoch):
                p = output_dir / f"{e}epoch"
                if p.exists() and e not in nbests:
                    shutil.rmtree(p)
                    _removed.append(str(p))
            if len(_removed) != 0:
                logging.info(
                    f"The snapshot was removed: " + ", ".join(_removed)
                )

            # 7. If any updating haven't happen, stops the training
            if all_steps_are_invalid:
                logging.warning(
                    f"The gradients at all steps are invalid "
                    f"in this epoch. Something seems wrong. "
                    f"This training was stopped at {iepoch}epoch"
                )
                break

            # 8. Check early stopping
            if patience is not None:
                _phase, _criterion, _mode = early_stopping_criterion
                if not reporter.has(_phase, _criterion):
                    raise RuntimeError(
                        f"{_phase}.{_criterion} is not found in stats: "
                        f"{reporter.get_all_keys()}"
                    )
                best_epoch, _ = reporter.sort_epochs_and_values(
                    _phase, _criterion, _mode
                )[0]
                if iepoch - best_epoch > patience:
                    logging.info(
                        f"[Early stopping] {_phase}.{_criterion} "
                        f"has not been improved "
                        f"{iepoch - best_epoch} epochs continuously. "
                        f"The training was stopped at {iepoch}epoch"
                    )
                    break

        else:
            logging.info(f"The training was finished at {max_epoch} epochs ")

        # 9. Average the n-best models
        cls.average_nbest_models(
            reporter=reporter,
            output_dir=output_dir,
            best_model_criterion=best_model_criterion,
            nbest=keep_n_best_snapshot,
        )

    @classmethod
    def train(
        cls,
        model: AbsE2E,
        iterator: DataLoader and Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        optimizer: torch.optim.Optimizer,
        reporter: SubReporter,
        scheduler: Optional[AbsBatchScheduler],
        ngpu: int,
        use_apex: bool,
        grad_noise: bool,
        accum_grad: int,
        grad_clip: float,
        log_interval: Optional[int],
        no_forward_run: bool,
        no_backward_run: bool,
    ) -> bool:
        assert check_argument_types()
        if log_interval is None:
            log_interval = max(len(iterator) // 20, 10)

        model.train()
        all_steps_are_invalid = True
        for iiter, (_, batch) in enumerate(iterator, 1):
            assert isinstance(batch, dict), type(batch)
            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                all_steps_are_invalid = False
                reporter.register({})
                continue

            if ngpu <= 1:
                # NOTE(kamo): data_parallel also should work with ngpu=1,
                # but for debuggability it's better to keep this block.
                loss, stats, weight = model(**batch)
            else:
                loss, stats, weight = data_parallel(
                    model, (), range(ngpu), module_kwargs=batch
                )
                # Weighted averaging of loss from torch-data-parallel
                loss = (loss * weight.to(loss.dtype)).sum(0) / weight.sum()
                stats = {
                    k: (v * weight.to(v.dtype)).sum(0) / weight.sum()
                    if v is not None
                    else None
                    for k, v in stats.items()
                }
                weight = weight.sum()
            reporter.register(stats, weight)

            if no_backward_run:
                all_steps_are_invalid = False
                continue

            if use_apex:
                from apex import amp

                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            del loss

            # gradient noise injection
            if grad_noise:
                add_gradient_noise(
                    model,
                    reporter.get_total_count(),
                    duration=100,
                    eta=1.0,
                    scale_factor=0.55,
                )

            # compute the gradient norm to check if it is normal or not
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip
            )
            if iiter % accum_grad == 0:
                if not np.isfinite(grad_norm):
                    logging.warning(
                        f"The grad norm is {grad_norm}. Skipping updating the model."
                    )
                else:
                    all_steps_are_invalid = False
                    optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

                # Register lr
                reporter.register(
                    {
                        f"lr_{i}": pg["lr"]
                        for i, pg in enumerate(optimizer.param_groups)
                        if "lr" in pg
                    },
                    not_increment_count=True,
                )

            if iiter % log_interval == 0:
                reporter.logging(nlatest=log_interval)
        return all_steps_are_invalid

    @classmethod
    @torch.no_grad()
    def eval(
        cls,
        model: AbsE2E,
        iterator: DataLoader and Iterable[Dict[str, torch.Tensor]],
        reporter: SubReporter,
        ngpu: int,
        no_forward_run: bool,
    ) -> None:
        assert check_argument_types()
        model.eval()
        for (_, batch) in iterator:
            assert isinstance(batch, dict), type(batch)
            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                reporter.register({})
                continue

            if ngpu <= 1:
                _, stats, weight = model(**batch)
            else:
                _, stats, weight = data_parallel(
                    model, (), range(ngpu), module_kwargs=batch
                )
                stats = {
                    k: (v * weight.to(v.dtype)).sum(0) / weight.sum()
                    if v is not None
                    else None
                    for k, v in stats.items()
                }
                weight = weight.sum()

            reporter.register(stats, weight)

    @classmethod
    @torch.no_grad()
    def plot_attention(
        cls,
        model: AbsE2E,
        output_dir: Path,
        iterator: DataLoader and Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        ngpu: int,
        reporter: SubReporter,
    ) -> None:
        assert check_argument_types()
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        model.eval()
        output_dir = Path(output_dir)
        for ids, batch in iterator:
            assert isinstance(batch, dict), type(batch)
            assert len(next(iter(batch.values()))) == len(ids), (
                len(next(iter(batch.values()))),
                len(ids),
            )
            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")

            # 1. Forwarding model and gathering all attentions
            #    calculate_all_attentions() uses single gpu only.
            att_dict = calculate_all_attentions(model, batch)

            # 2. Plot attentions: This part is slow due to matplotlib
            for k, att_list in att_dict.items():
                assert len(att_list) == len(ids), (len(att_list), len(ids))
                for id_, att_w in zip(ids, att_list):

                    if isinstance(att_w, torch.Tensor):
                        att_w = att_w.detach().cpu().numpy()

                    if att_w.ndim == 2:
                        att_w = att_w[None]
                    elif att_w.ndim > 3 or att_w.ndim == 1:
                        raise RuntimeError(
                            f"Must be 2 or 3 dimension: {att_w.ndim}"
                        )

                    w, h = plt.figaspect(1.0 / len(att_w))
                    fig = plt.Figure(figsize=(w * 1.3, h * 1.3))
                    axes = fig.subplots(1, len(att_w))
                    if len(att_w) == 1:
                        axes = [axes]

                    for ax, aw in zip(axes, att_w):
                        ax.imshow(aw.astype(np.float32), aspect="auto")
                        ax.set_title(f"{k}_{id_}")
                        ax.set_xlabel("Input")
                        ax.set_ylabel("Output")
                        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

                    p = output_dir / id_ / (k + ".png")
                    p.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(p)

                    # Dummy register() stimulates to increment the counter
                    reporter.register({})

    @classmethod
    @torch.no_grad()
    def average_nbest_models(
        cls,
        output_dir: Path,
        reporter: Reporter,
        best_model_criterion: Sequence[Sequence[str]],
        nbest: int,
    ) -> None:
        assert check_argument_types()
        # 1. Get nbests: List[Tuple[str, str, List[Tuple[epoch, value]]]]
        nbest_epochs = [
            (ph, k, reporter.sort_epochs_and_values(ph, k, m)[:nbest])
            for ph, k, m in best_model_criterion
            if reporter.has(ph, k)
        ]

        _loaded = {}
        for ph, cr, epoch_and_values in nbest_epochs:
            # Note that len(epoch_and_values) doesn't always equal to nbest.

            op = output_dir / f"{ph}.{cr}.ave_{len(epoch_and_values)}best.pt"
            logging.info(
                f"Averaging {len(epoch_and_values)}best models: "
                f'criterion="{ph}.{cr}": {op}'
            )

            avg = None
            # 2.a Averaging model
            for e, _ in epoch_and_values:
                if e not in _loaded:
                    _loaded[e] = torch.load(
                        output_dir / f"{e}epoch" / "model.pt",
                        map_location="cpu",
                    )
                states = _loaded[e]

                if avg is None:
                    avg = states
                else:
                    # Accumulated
                    for k in avg:
                        avg[k] += states[k]
            for k in avg:
                avg[k] /= len(epoch_and_values)

            # 2.b Save the ave model and create a symlink
            torch.save(avg, op)
            sym_op = output_dir / f"{ph}.{cr}.ave.pt"
            if sym_op.is_symlink() or sym_op.exists():
                sym_op.unlink()
            sym_op.symlink_to(op.name)
