from gym.envs.registration import register
register(
    id='Franka-PCB-v0',
    entry_point='robot_infra.env:PCBEnv',
)
register(
    id='Franka-RouteCable-v0',
    entry_point='robot_infra.env:RouteCableEnv',
)
register(
    id='Franka-ResetCable-v0',
    entry_point='robot_infra.env:ResetCableEnv',
)
register(
    id='Franka-BinPick-v0',
    entry_point='robot_infra.env:BinPickEnv',
)