import pandas as pd
import argparse
import json
import numpy as np

from sklearn.metrics import roc_curve

def compute_eer(label, pred, positive_label=1):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = roc_curve(label, pred)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer



def main(private_test_url, private_prediction_url):
    gt_label_column = "label"
    sub_sim_column = "similarity"
    id_column = "pair_id"

    sub_df = pd.read_csv(private_prediction_url)
    gt_df = pd.read_csv(private_test_url)

    gt_df = gt_df.astype({id_column: int})
    sub_df = sub_df.astype({id_column: int})

    gt_df = gt_df.join(sub_df.set_index(id_column), on=id_column, how="left")

    if gt_df[sub_sim_column].isna().any():
        print("Не все `pair_id` присутствуют в сабмите")

    y_score = sub_df[sub_sim_column].tolist()
    y_true = gt_df[gt_label_column].tolist()

    return compute_eer(y_true, y_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--public_test_url", type=str, required=True)
    parser.add_argument("--public_prediction_url", type=str, required=True)
    parser.add_argument("--private_test_url", type=str, required=False)
    parser.add_argument("--private_prediction_url", type=str, required=False)
    args = parser.parse_args()
    public_score = main(args.public_test_url, args.public_prediction_url)

    private_score = None
    if args.private_test_url and args.private_prediction_url:
        private_score = main(args.private_test_url, args.private_prediction_url)

    print(
        json.dumps(
            {
                "public_score": public_score,
                "private_score": private_score,
            }
        )
    )
