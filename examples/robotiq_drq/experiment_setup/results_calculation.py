from os.path import join, exists, abspath
import numpy as np
import pandas as pd
import pathlib
import pickle as pkl

np.set_printoptions(precision=3, suppress=True)
pd.set_option('display.max_columns', 12, 'display.max_rows', 11)

wdir = pathlib.Path().resolve()

eval10_files = {
    '1 BT': 'BT/trajectories 08-23 1834.pkl',
    '2 BC': 'BC/trajectories 08-23 1853.pkl',
    '3 SAC': 'SAC/trajectories 08-27 1833.pkl',
    # '4 DRQ RGB': 'ResNet10/trajectories 08-23 1821.pkl',
    '5 DRQ RGB': 'ResNet18/trajectories 08-23 1807.pkl',
    '6 DRQ Depth': 'Depth Image/trajectories 08-27 1825.pkl',
    '7 DRQ Voxel': 'VoxNet/trajectories 08-24 1423.pkl',
    '8 DRQ Voxel': 'VoxNet_pretrained/trajectories 08-24 1451.pkl',
    '9 DRQ Voxel': 'VoxNet_1quat/trajectories 08-24 1442.pkl',
    '10 DRQ Voxel': 'VoxNet_pretrained_1quat/trajectories 08-24 1504.pkl',
    # '11 DRQ Voxel': 'VoxNet_pretrained_gripper_1quat/trajectories 08-29 1536.pkl',
}

eval5_files = {
    '1 BT': 'BT/trajectories 08-28 1734.pkl',
    '2 BC': 'BC/trajectories 08-28 1744.pkl',
    '3 SAC': 'SAC/trajectories 08-28 1756.pkl',
    # '4 DRQ RGB': 'ResNet10/trajectories 08-28 1815.pkl',
    '5 DRQ RGB': 'ResNet18/trajectories 08-28 1805.pkl',
    '6 DRQ Depth': 'Depth Image/trajectories 08-28 1823.pkl',
    '7 DRQ Voxel': 'VoxNet/trajectories 08-28 1830.pkl',
    '8 DRQ Voxel': 'VoxNet_pretrained/trajectories 08-29 1037.pkl',
    '9 DRQ Voxel': 'VoxNet_1quat/trajectories 08-29 1024.pkl',
    '10 DRQ Voxel': 'VoxNet_pretrained_1quat/trajectories 08-29 1048.pkl',
    # '11 DRQ Voxel': 'VoxNet_pretrained_gripper_1quat/trajectories 08-29 1544.pkl',
    '12 DRQ Voxel pqt': 'VoxNet_pretrained_1quat/trajectories temp_ens 08-30 1037.pkl',
    '13 DRQ Voxel pqgt': 'VoxNet_pretrained_gripper_1quat/trajectories temp_ens 08-30 1028.pkl',
}


def create_df(files):
    infos = {"Policy": [' '.join(name.split(' ')[1:]) for name in files.keys()],
             "Success": np.zeros((len(files.keys()))),
             "Reward": np.zeros((len(files.keys()))),
             "Rd": np.zeros((len(files.keys()))),
             # "A": np.zeros((len(files.keys()))),
             "Time": np.zeros((len(files.keys()))),
             "Td": np.zeros((len(files.keys()))),
             }

    for i, (name, file) in enumerate(files.items()):
        assert exists(file)
        with open(file, 'rb') as f:
            trajectories = pkl.load(f)

        # print(f"\n---{name} ({file.split('/')[0]}):")
        # print(f"{len(trajectories)} trajectories --> {[len(t['traj']) for t in trajectories]}")

        success = np.asarray([traj['success'] for traj in trajectories])
        time = np.asarray([traj['time'] for traj in trajectories])
        running_reward = np.asarray([np.sum(np.asarray([t["rewards"] for t in traj['traj']])) for traj in trajectories])
        # print(f"success rate {np.mean(success) * 100:.1f}  time: {np.mean(time):.2f}")
        # print(
        #     f"reward: {np.mean(running_reward):.2f}   where successful: {np.nanmean(np.where(success, running_reward, np.nan)):.2f}")

        actions = [np.asarray([t['actions'][:6] for t in traj["traj"]]) for traj in
                   trajectories]  # 30 x shape (step, action_dim)
        action_diff = np.asarray(
            [np.linalg.norm(np.diff(a, axis=0), ord=2, axis=0) for a in actions])  # shape (traj, action_dim)
        print(f"{name}  action diff: {np.mean(action_diff, axis=0)}   {np.mean(action_diff):.2f}")
        if type(trajectories[0]["traj"][0]["observations"]) == dict:
            velocities = [[t["observations"]["state"].reshape(-1)[21:] for t in traj["traj"]] for traj in
                          trajectories]
            pos = [[t["observations"]["state"].reshape(-1)[12:18] for t in traj["traj"]] for traj in
                          trajectories]
        else:
            velocities = [[t["observations"].reshape(-1)[21:] for t in traj["traj"]] for traj in
                          trajectories]
            pos = [[t["observations"].reshape(-1)[12:18] for t in traj["traj"]] for traj in
                      trajectories]

        # plen = np.asarray([np.sum(np.abs(np.diff(p, axis=0))) for p in pos])
        # vel_p = np.asarray([np.linalg.norm(np.diff(p, axis=0), ord=2, axis=0) for p in pos])
        # vel = np.asarray([np.linalg.norm(vel, ord=2, axis=0) for vel in velocities])
        # accel = np.asarray([np.linalg.norm(np.diff(vel, axis=0), ord=2, axis=0) for vel in velocities])
        # print(f"plen {np.mean(plen):.3f}")
        # print(f"{name}  accel diff: {np.mean(accel, axis=0)}   {np.mean(accel):.2f}   {np.mean(vel):.2f} {np.mean(vel_p)*10:.2f}")

        infos['Success'][i] = success.mean() * 100.
        infos['Reward'][i] = running_reward.mean()
        infos['Rd'][i] = np.nanstd(np.where(success, running_reward, np.nan))
        infos['Time'][i] = time.mean()
        infos['Td'][i] = np.nanstd(np.where(success, time, np.nan))
        # infos['A'][i] = action_diff.mean()

    return pd.DataFrame(infos)


def f2s(x, precision=1):
    # return string of a float with given precision, without zeros
    return f"{float(x):.{precision}f}".rstrip('0').rstrip('.')


def highlight_extremes(df, max_subset, min_subset):
    # Highlight maximum in max_subset
    for column in max_subset:
        max_value = df[column].max()
        df[column] = df[column].apply(lambda x: f"\\bfseries {f2s(x)}" if x == max_value else f2s(x))

    # Highlight minimum in min_subset
    for column in min_subset:
        min_value = df[column].min()
        df[column] = df[column].apply(lambda x: f"\\bfseries {f2s(x, 2)}" if x == min_value else f2s(x, 2))

    for column in df.columns:
        if column in ["Rd", "Td", "Rd5", "Td5"]:
            df[column] = df[column].apply(lambda x: f"$\\pm{f2s(x)}$")

    return df


def same_strlen(df):                # TODO does not work (textbf...)
    for column in df.columns[1:]:
        max_len = np.max([len(s) for s in df[column]])
        print(max_len)
        df[column] = df[column].apply(lambda x: " "*(max_len-len(x)) + x)
    return df


if __name__ == '__main__':
    box_10_df = create_df(eval10_files)
    box_5_df = create_df(eval5_files)

    box_5_df.rename(columns={"Success": "Success5", "Reward": "Reward5", "Time": "Time5", "Rd": "Rd5", "Td": "Td5"},
                    inplace=True)
    box_5_df = box_5_df.drop(columns=box_5_df.columns[0])

    combined = box_10_df.merge(box_5_df, how='inner', left_index=True, right_index=True)

    print("\nEvaluation on seen boxes\n", box_10_df)
    print("\nEvaluation on unseen boxes\n", box_5_df)
    print("\nboth:\n", combined)

    max = np.array([1, 2, 6, 7])
    min = np.array([4, 9, ])
    df = highlight_extremes(combined, max_subset=combined.columns[max], min_subset=combined.columns[min])
    df = df.rename(index={i: f"({i + 1})" for i in list(df.index)})

    styled = (
        df.style
        # .hide(axis="index")
        .format(decimal='.', thousands='\'', precision=2)
    )
    latex = styled.to_latex(hrules=True)
    print("\n\t", latex.replace("\n", "\n\t\t"))
