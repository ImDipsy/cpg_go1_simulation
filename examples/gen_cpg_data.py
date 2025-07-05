import multiprocessing
import time

import tqdm

from cpg_go1_simulation.stein.base import Args
from cpg_go1_simulation.stein.implementations import CPG8Neuron


def my_handle(args: Args):
    """Execute CPG simulation based on args"""

    cpg_classes = {
        "8neuron": CPG8Neuron,
    }

    CPGClass = cpg_classes[args.neuron_type]

    signal = CPGClass(
        before_ftype=args.ftype[0],
        after_ftype=args.ftype[1],
        total_time=args.total_time,
        toc=args.toc,
        _if_backward=args._if_backward,
        _if_mlr_perturbation=args._if_mlr_perturbation,
        _if_state_perturbation=args._if_state_perturbation,
    )

    signal.export_csv()


def gen_groups():
    """Generate parameter groups for processing"""
    ftype_list = [
        # [1, 1],
        [2, 2],
        # [3, 3],
        # [4, 4],
        # [5, 5],
        # [1, 2],
    ]

    neuron_types = [
        "8neuron",
        # "8neuron_backward",
        # "10neuron",
        # "12neuron",
    ]

    for ftype in ftype_list:
        for neuron_type in neuron_types:
            # if you want to generate five gaits, uncomment the following line
            yield Args(ftype=ftype, total_time=1, neuron_type=neuron_type)


def main():
    start_time = time.time()

    with multiprocessing.Pool() as pool:
        list(
            tqdm.tqdm(
                pool.imap(my_handle, gen_groups()),
                total=2,
            )
        )

    print(f"Execution time: {time.time() - start_time:.4f} seconds")


if __name__ == "__main__":
    main()
