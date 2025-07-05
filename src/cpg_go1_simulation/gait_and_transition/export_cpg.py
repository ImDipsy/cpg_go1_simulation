from cpg_go1_simulation.config import GAIT_MAP
from cpg_go1_simulation.stein.implementations import CPG8Neuron


def export_cpg_data(
    gait_type: str = "walk",
    command_time: float = 15.0,
    total_time: float = 25.0,
    _if_backward: bool = False,
):
    """Generate cpg data for the gait or transition type.

    Args:
        gait_type (str): Can be either a single gait (e.g., 'walk') or a transition gait (e.g., 'walk_to_trot')
        command_time (Optional[float]): Time to switch from the first gait to the second gait. Defaults to 15.0.
        total_time (Optional[float]): Total time for the gait. Defaults to 25.0.
    Returns:
        float: Execution time of the gait transition
    """
    if "_to_" in gait_type:
        # Handle transition gait
        before_gait, after_gait = gait_type.split("_to_")
        before_gait_id = GAIT_MAP[before_gait]
        after_gait_id = GAIT_MAP[after_gait]
    else:
        gait_id = GAIT_MAP[gait_type]
        before_gait_id = after_gait_id = gait_id

    signal = CPG8Neuron(
        before_ftype=before_gait_id,
        after_ftype=after_gait_id,
        total_time=total_time,
        toc=command_time,
        _if_backward=_if_backward,
    )
    signal.export_csv()

    return signal.real_toc if "_to_" in gait_type else None


if __name__ == "__main__":
    export_cpg_data()
