import argparse
import os
import numpy as np

def load_data(npz_filename):
    # 加载 .npz 文件并读取数据
    with np.load(npz_filename) as data:
        return data['data']  # 假设数据集名为 'data'

def generate_data_splits(data, seq_len, horizon):
    num_samples, num_nodes, num_features = data.shape
    x, y = [], []
    for i in range(num_samples - seq_len - horizon + 1):
        x.append(data[i:(i + seq_len), :, :])
        y.append(data[(i + seq_len):(i + seq_len + horizon), :, :])
    return np.array(x), np.array(y)


def generate_train_val_test(args):
    # Load data from NPZ file specified by the user
    data = load_data(args.traffic_df_filename)

    # Generate data splits according to user-defined seq_len and horizon
    x, y = generate_data_splits(data, args.seq_len, args.horizon)

    # Define the proportion of the data splits
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.6)
    num_val = num_samples - num_test - num_train

    # Split the data into training, validation, and testing
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x[num_train:num_train + num_val], y[num_train:num_train + num_val]
    x_test, y_test = x[-num_test:], y[-num_test:]
    print(x_train.shape)

    # Save each dataset split into respective .npz file
    for cat, _x, _y in zip(["train", "val", "test"], [x_train, x_val, x_test], [y_train, y_val, y_test]):
        print(f"{cat} x: {_x.shape}, y: {_y.shape}")
        filename = f"{cat}.npz"
        np.savez_compressed(
            os.path.join(args.output_dir, filename),
            x=_x,
            y=_y
        )
        
        


def main(args):
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    # Process and save the data splits
    generate_train_val_test(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="data/PEMS04", help="Output directory."
    )
    parser.add_argument(
        "--traffic_df_filename",
        type=str,
        default="data/PEMS04/pems04.npz",
        help="Raw traffic readings in NPZ format.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=12,
        help="Length of the input sequence."
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=12,
        help="Length of the output sequence."
    )

    args = parser.parse_args()
    main(args)
