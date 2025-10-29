import random
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def get_representative_value(values, method='min'):
    """
    Get a representative value from a list that may contain outliers.

    Parameters:
    -----------
    values : list of float.
    method : str, choice from 'median', 'trimmed_mean', 'iqr', 'mad', 'zscore'.
    
    Returns:
    --------
    float
    """
    values = np.array(values)

    if method == 'min':
        return np.min(values)

    elif method == 'min_avg':
        # Take the mean of the smallest N values
        n = max(10, len(values) // 5)  # At least 10, or 20% of the total count
        n = min(n, len(values))  # Not exceed the total count
        sorted_values = np.sort(values)
        return np.mean(sorted_values[:n])

    elif method == 'percentile':
        # Take the lower percentile (e.g. 5% percentile)
        return np.percentile(values, 5)

    elif method == 'trimmed_min':
        # Remove the smallest 1 extreme value, then take the mean of the remaining smaller values
        # (to prevent an abnormal low value)
        sorted_values = np.sort(values)
        n = max(10, len(values) // 5)
        n = min(n, len(values) - 1)
        return np.mean(sorted_values[1:n+1])

    elif method == 'median':
        # Directly use the median (most robust to outliers)
        return np.median(values)

    elif method == 'trimmed_mean':
        # Trimmed mean - remove the highest and lowest 10% of values
        return stats.trim_mean(values, proportiontocut=0.1)

    elif method == 'iqr':
        # IQR method - remove outliers beyond the interquartile range
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered = values[(values >= lower_bound) & (values <= upper_bound)]
        return np.mean(filtered) if len(filtered) > 0 else np.median(values)

    elif method == 'mad':
        # MAD (Median Absolute Deviation) - more robust method
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        threshold = 3  # Adjustable
        filtered = values[np.abs(values - median) <= threshold * mad]
        return np.mean(filtered) if len(filtered) > 0 else median

    elif method == 'zscore':
        # Z-score method - remove values beyond N standard deviations
        mean = np.mean(values)
        std = np.std(values)
        threshold = 2  # Adjustable (2 or 3 standard deviations)
        filtered = values[np.abs(values - mean) <= threshold * std]
        return np.mean(filtered) if len(filtered) > 0 else mean

    else:
        raise ValueError(f"Unknown method: {method}")


def fit_linear_model(data_dict):
    """
    Fit a linear model: a*x_2 + b*x + c*x_s_2 + d*x_s + e*n + f*n_s + g = y

    Parameters:
    -----------
    data_dict : dict
        Format: {(x_2, x, x_s_2, x_s, n, n_s): y}

    Returns:
    --------
    dict : Contains the fitting results and metrics
    """
    # Extract data
    keys = list(data_dict.keys())
    values = list(data_dict.values())

    # Construct the feature matrix X and target vector y
    x = np.array([[x_2 / 1024 / 1024, x / 1024, x_s_2 / 1024 / 1024, x_s / 1024, n, n_s] for x_2, x, x_s_2, x_s, n, n_s in keys])  # shape: (n_samples, 6)
    y = np.array(values)  # shape: (n_samples,)

    # Fit the linear model
    model = LinearRegression()
    model.fit(x, y)

    # Get the parameters
    a, b, c, d, e, f = model.coef_
    g = model.intercept_

    # Predict
    y_pred = model.predict(x)

    # Calculate various metrics
    r2 = r2_score(y, y_pred)  # R² coefficient (the closer to 1, the better)
    mse = mean_squared_error(y, y_pred)  # Mean squared error
    rmse = np.sqrt(mse)  # Root mean squared error
    mae = mean_absolute_error(y, y_pred)  # Mean absolute error

    # Calculate relative error
    mape = np.mean(np.abs((y - y_pred) / y)) * 100  # Mean absolute percentage error

    # Calculate maximum error
    max_error = np.max(np.abs(y - y_pred))
    max_relative_error = np.max(np.abs((y - y_pred) / y)) * 100

    return {
        'coefficients': {'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'f': f, 'g': g},
        'metrics': {
            'R²': r2,  # The closer to 1, the better, >0.9 means good fitting
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE(%)': mape,  # Mean absolute percentage error
            'Max_Error': max_error,
            'Max_Relative_Error(%)': max_relative_error
        },
    }


if __name__ == "__main__":

    fb = "forward"
    threshold_of_short_seq = 0

    seqlens_per_rank = [[] for _ in range(8)]
    times_per_rank = [[] for _ in range(8)]
    for rank in range(8):
        with open(f"profiling_results/seqlens_dp{rank}.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                seqlens_per_rank[rank].append([int(x) for x in line.strip().split(" ")])
        with open(f"profiling_results/L_{fb}_dp{rank}.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                times_per_rank[rank].append([float(x) for x in line.strip().split(" ")])

    # Avoid num_items of seqlens > tiem
    for rank in range(8):
        l = min(len(seqlens_per_rank[rank]), len(times_per_rank[rank]))
        seqlens_per_rank[rank] = seqlens_per_rank[rank][:l]
        times_per_rank[rank] = times_per_rank[rank][:l]

    # Merge samples with same x^2 and x
    samples = {}
    sample_to_raw = {}
    for rank in range(8):
        seqlens = seqlens_per_rank[rank]
        times = times_per_rank[rank]
        for seqlen, time in zip(seqlens, times):
            x_2 = sum([x*x for x in seqlen])
            x = sum(seqlen)
            x_s_2 = sum([x*x for x in seqlen if x < threshold_of_short_seq])
            x_s = sum([x for x in seqlen if x < threshold_of_short_seq])
            n = len(seqlen)
            n_s = len([x for x in seqlen if x < threshold_of_short_seq])
            key = (x_2, x, x_s_2, x_s, n, n_s)
            if key not in samples:
                samples[key] = []
                sample_to_raw[key] = []
            samples[key].extend(time)
            sample_to_raw[key].append((seqlen, time))

    # Get representative values for each sample
    for key in samples:
        samples[key] = get_representative_value(samples[key], method='min')

    # Print the first 3 micro batches
    # seqlens = [
    #     [4971, 4901, 2402, 1069, 871, 379],
    #     [13760, 591, 266],
    #     [8589, 5275, 4737, 2540, 1561, 1372, 679, 502],
    # ]
    # for seqlen in seqlens:
    #     x_2 = sum([x*x for x in seqlen])
    #     x = sum(seqlen)
    #     x_s = sum([x for x in seqlen if x < threshold_of_short_seq])
    #     x_s_2 = sum([x*x for x in seqlen if x < threshold_of_short_seq])
    #     n = len(seqlen)
    #     n_s = len([x for x in seqlen if x < threshold_of_short_seq])
    #     print(samples[(x_2, x, x_s_2, x_s, n, n_s)])

    sample_items = list(samples.items())
    num_train_samples = len(sample_items) // 2
    train_samples = dict(sample_items[:num_train_samples])
    valid_samples = dict(sample_items[num_train_samples:])

    result = fit_linear_model(train_samples)
    print(result["coefficients"])
    print(result["metrics"])

    a = result["coefficients"]["a"]
    b = result["coefficients"]["b"]
    c = result["coefficients"]["c"]
    d = result["coefficients"]["d"]
    e = result["coefficients"]["e"]
    f = result["coefficients"]["f"]
    g = result["coefficients"]["g"]

    num_2 = 0
    num_5 = 0
    num_10 = 0

    for i, (key, value) in enumerate(valid_samples.items()):
        x_2, x, x_s_2, x_s, n, n_s = key
        x_2 = x_2 / 1024 / 1024
        x = x / 1024
        x_s_2 = x_s_2 / 1024 / 1024
        x_s = x_s / 1024
        estimated = a * x_2 + b * x + c * x_s_2 + d * x_s + e * n + f * n_s + g
        error = (estimated - value) / value * 100

        if abs(error) < 2:
            num_2 += 1
        if abs(error) < 5:
            num_5 += 1
        if abs(error) < 10:
            num_10 += 1

        # if abs(error) > 2:
        #     print("\n")
        #     print("measured: ", value)
        #     print("estimated: ", estimated)
        #     print("error: ", error, "%")
        #     print("\n")
        #     for seqlen, time in sample_to_raw[key]:
        #         print(seqlen, time)
        #         times.extend(time)
        #     input("\nInput any key to continue...")

    total = len(valid_samples)
    print(f"error < 2%: {num_2 / total * 100}%, number of error > 2%: {total - num_2}, total: {total}")
    print(f"error < 5%: {num_5 / total * 100}%, number of error > 5%: {total - num_5}, total: {total}")
    print(f"error < 10%: {num_10 / total * 100}%, number of error > 10%: {total - num_10}, total: {total}")
