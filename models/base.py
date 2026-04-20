import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict, namedtuple
from tqdm import tqdm

from .loss import l1_l2_loss

import time


def add_average(df: pd.DataFrame, gp=False):
    """Append per-run and overall averages to a results dataframe."""
    averages_df = pd.DataFrame()
    run_number_list = np.array(range(1, 3))
    for i in run_number_list:
        run_df = df[df["run_number"] == i]
        if not run_df.empty:
            new_row = {
                "year": f"Average_{i}",
                "run_number": i,
                "time_idx": "",
                "RMSE": run_df["RMSE"].mean(),
                "ME": run_df["ME"].mean(),
                "R_2": run_df["R_2"].mean(),
            }
            if gp:
                new_row["ME_GP"] = run_df["ME_GP"].mean()
                new_row["RMSE_GP"] = run_df["RMSE_GP"].mean()
                new_row["R_2_GP"] = run_df["R_2_GP"].mean()
            averages_df = pd.concat(
                [averages_df, pd.DataFrame([new_row])], ignore_index=True
            )

    new_row = {
        "year": "Average",
        "run_number": "",
        "time_idx": "",
        "RMSE": df["RMSE"].mean(),
        "ME": df["ME"].mean(),
        "R_2": df["R_2"].mean(),
    }
    if gp:
        new_row["ME_GP"] = df["ME_GP"].mean()
        new_row["RMSE_GP"] = df["RMSE_GP"].mean()
        new_row["R_2_GP"] = df["R_2_GP"].mean()
    averages_df = pd.concat(
        [averages_df, pd.DataFrame([new_row])], ignore_index=True
    )

    return pd.concat([df, averages_df], ignore_index=True)


class ModelBase:
    """Base class for DFYP training and evaluation."""

    def __init__(
        self,
        model,
        model_type,
        savedir,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        """Initialize the wrapped model and output directory."""
        self.savedir = Path(savedir)
        self.savedir.mkdir(parents=True, exist_ok=True)
        print(self.savedir)

        print(f"Using {device.type}")
        if device.type != "cpu":
            model = model.cuda()
        self.model = model
        self.model_type = model_type

        self.device = device

        # Keep initialization deterministic across runs.
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    def run(
        self,
        path_to_histogram=Path("data/img_output/histogram_all_full.npz"),
        times="all",
        pred_years=None,
        num_runs=2,
        train_steps=25000,
        batch_size=32,
        starter_learning_rate=1e-3,
        weight_decay=0,
        l1_weight=0,
        patience=10,
    ):
        """Train models across prediction years and save aggregated results."""

        with np.load(path_to_histogram) as hist:
            images = hist["output_image"]
            locations = hist["output_locations"]
            yields = hist["output_yield"]
            years = hist["output_year"]
            indices = hist["output_index"]

        years_list, run_numbers, rmse_list, me_list, mae_list, times_list, all_time, r_2_list = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        if pred_years is None:
            pred_years = range(2009, 2016)
        elif isinstance(pred_years, int):
            pred_years = [pred_years]

        if times == "all":
            times = [32]
        else:
            times = range(12, 32, 4)

        for pred_year in pred_years:
            for run_number in range(1, num_runs + 1):
                for s in times:
                    print(f"Training to predict on {pred_year}, Run number {run_number}")
                    begin_time = time.time()
                    results = self._run_1_year(
                        images,
                        yields,
                        years,
                        locations,
                        indices,
                        pred_year,
                        s,
                        run_number,
                        train_steps,
                        batch_size,
                        starter_learning_rate,
                        weight_decay,
                        l1_weight,
                        patience,
                    )
                    end_time = time.time()

                    years_list.append(pred_year)
                    run_numbers.append(run_number)
                    times_list.append(s)
                    all_time.append(end_time - begin_time)

                    rmse, me, mae, r_2 = results
                    rmse_list.append(rmse)
                    me_list.append(me)
                    mae_list.append(mae)
                    r_2_list.append(r_2)
                print("-----------")

        data = {
            "year": years_list,
            "run_number": run_numbers,
            "time_idx": times_list,
            "RMSE": rmse_list,
            "ME": me_list,
            "MAE": mae_list,
            "R_2": r_2_list,
            "time": all_time,
        }
        results_df = pd.DataFrame(data=data)
        results_df = add_average(results_df, gp=False)
        results_df.to_csv(self.savedir / "results.csv", index=False)

    def _run_1_year(
        self,
        images,
        yields,
        years,
        locations,
        indices,
        predict_year,
        time,
        run_number,
        train_steps,
        batch_size,
        starter_learning_rate,
        weight_decay,
        l1_weight,
        patience,
    ):
        """Train one model for one prediction year and save its checkpoint."""
        train_data, test_data = self.prepare_arrays(
            images, yields, locations, indices, years, predict_year, time
        )

        self.reinitialize_model(time=time)

        train_scores, val_scores = self._train(
            train_data.images,
            train_data.yields,
            train_data.years,
            train_steps,
            batch_size,
            starter_learning_rate,
            weight_decay,
            l1_weight,
            patience,
        )

        results = self._predict(*train_data, *test_data, batch_size)

        model_information = {
            "state_dict": self.model.state_dict(),
            "val_loss": val_scores["loss"],
            "train_loss": train_scores["loss"],
        }
        for key in results:
            model_information[key] = results[key]

        filename = f"{predict_year}_{run_number}_{time}.pth"
        torch.save(model_information, self.savedir / filename)
        return self.analyze_results(
            model_information["test_real"],
            model_information["test_pred"],
        )

    def _train(
        self,
        train_images,
        train_yields,
        train_years,
        train_steps,
        batch_size,
        starter_learning_rate,
        weight_decay,
        l1_weight,
        patience,
    ):
        """Run the training loop with a held-out validation split."""

        total_size = train_images.shape[0]
        val_size = total_size // 10
        train_size = total_size - val_size

        print(
            f"After split, training on {train_size} examples, "
            f"validating on {val_size} examples"
        )

        train_dataset, val_dataset = random_split(
            TensorDataset(train_images, train_yields, train_years),
            (train_size, val_size),
            torch.Generator().manual_seed(0),
        )
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        optimizer = torch.optim.Adam(
            [pam for pam in self.model.parameters()],
            lr=starter_learning_rate,
            weight_decay=weight_decay,
        )
        num_epochs = int(train_steps / (train_images.shape[0] / batch_size))
        print(f"Training for {num_epochs} epochs")
        train_scores = defaultdict(list)
        val_scores = defaultdict(list)

        step_number = 0
        min_loss = np.inf
        best_state = self.model.state_dict()
        if patience is not None:
            epochs_without_improvement = 0
        print(self.model_type)
        for epoch in range(num_epochs):
            self.model.train()

            running_train_scores = defaultdict(list)
            for train_x, train_y, train_year in tqdm(train_dataloader):
                optimizer.zero_grad()
                model_output = self._forward_model(train_x, train_year)
                pred_y = model_output
                loss, running_train_scores = l1_l2_loss(
                    pred_y, train_y, l1_weight, running_train_scores
                )
                loss.backward()
                optimizer.step()

                train_scores["loss"].append(loss.item())
                step_number += 1

                if step_number in [4000, 20000]:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] /= 10

            train_output_strings = []
            for key, val in running_train_scores.items():
                train_output_strings.append(
                    "{}: {}".format(key, round(np.array(val).mean(), 5))
                )
            running_val_scores = defaultdict(list)
            self.model.eval()
            with torch.no_grad():
                for val_x, val_y, val_year in tqdm(val_dataloader):
                    model_output = self._forward_model(val_x, val_year)
                    val_pred_y = model_output
                    val_loss, running_val_scores = l1_l2_loss(
                        val_pred_y, val_y, l1_weight, running_val_scores
                    )

                    val_scores["loss"].append(val_loss.item())

            val_output_strings = []
            for key, val in running_val_scores.items():
                val_output_strings.append(
                    "{}: {}".format(key, round(np.array(val).mean(), 5))
                )

            print("TRAINING: {}".format(", ".join(train_output_strings)))
            print("VALIDATION: {}".format(", ".join(val_output_strings)))

            epoch_val_loss = np.array(running_val_scores["loss"]).mean()
            if epoch_val_loss < min_loss:
                best_state = self.model.state_dict()
                min_loss = epoch_val_loss

                if patience is not None:
                    epochs_without_improvement = 0
            elif patience is not None:
                epochs_without_improvement += 1

                if epochs_without_improvement == patience:
                    self.model.load_state_dict(best_state)
                    print("Early stopping!")
                    break

        self.model.load_state_dict(best_state)
        return train_scores, val_scores

    def _predict(
        self,
        train_images,
        train_yields,
        train_locations,
        train_indices,
        train_years,
        test_images,
        test_yields,
        test_locations,
        test_indices,
        test_years,
        batch_size,
    ):
        """Run inference on the train and test splits."""
        train_dataset = TensorDataset(
            train_images, train_yields, train_locations, train_indices, train_years
        )

        test_dataset = TensorDataset(
            test_images, test_yields, test_locations, test_indices, test_years
        )

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        results = defaultdict(list)

        self.model.eval()
        with torch.no_grad():
            for train_im, train_yield, train_loc, train_idx, train_year in tqdm(
                train_dataloader
            ):
                model_output = self._forward_model(
                    train_im,
                    train_year,
                    return_last_dense=False,
                )
                pred = model_output
                results["train_pred"].extend(pred.squeeze(1).tolist())
                results["train_real"].extend(train_yield.squeeze(1).tolist())
                results["train_loc"].append(train_loc.numpy())
                results["train_indices"].append(train_idx.numpy())
                results["train_years"].extend(train_year.tolist())

            for test_im, test_yield, test_loc, test_idx, test_year in tqdm(
                test_dataloader
            ):
                model_output = self._forward_model(
                    test_im,
                    test_year,
                    return_last_dense=False,
                )
                pred = model_output
                results["test_pred"].extend(pred.squeeze(1).tolist())
                results["test_real"].extend(test_yield.squeeze(1).tolist())
                results["test_loc"].append(test_loc.numpy())
                results["test_indices"].append(test_idx.numpy())
                results["test_years"].extend(test_year.tolist())

        for key in results:
            if key in [
                "train_loc",
                "test_loc",
                "train_indices",
                "test_indices",
                "train_operator_weights",
                "test_operator_weights",
            ]:
                results[key] = np.concatenate(results[key], axis=0)
            else:
                results[key] = np.array(results[key])
        return results

    def _predict_gr(
        self,
        test_images,
        test_yields,
        test_locations,
        test_indices,
        test_years,
        batch_size,
    ):
        """Run inference on a test split only."""
        test_dataset = TensorDataset(
            test_images, test_yields, test_locations, test_indices, test_years
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        results = defaultdict(list)

        self.model.eval()
        with torch.no_grad():
            for test_im, test_yield, test_loc, test_idx, test_year in tqdm(
                test_dataloader
            ):
                model_output = self._forward_model(
                    test_im,
                    test_year,
                    return_last_dense=False,
                )
                pred = model_output
                results["test_pred"].extend(pred.squeeze(1).tolist())
                results["test_real"].extend(test_yield.squeeze(1).tolist())
                results["test_loc"].append(test_loc.numpy())
                results["test_indices"].append(test_idx.numpy())
                results["test_years"].extend(test_year.tolist())

        for key in results:
            if key in [
                "test_loc",
                "test_indices",
            ]:
                results[key] = np.concatenate(results[key], axis=0)
            else:
                results[key] = np.array(results[key])
        return results

    def prepare_arrays(
        self, images, yields, locations, indices, years, predict_year, time
    ):
        """Split data by year, normalize inputs, and convert arrays to tensors."""
        train_idx = np.nonzero(years < predict_year)[0]
        test_idx = np.nonzero(years == predict_year)[0]

        train_images, test_images = self._normalize(
            images[train_idx], images[test_idx]
        )

        print(
            f"Train set size: {train_idx.shape[0]}, Test set size: {test_idx.shape[0]}"
        )

        Data = namedtuple(
            "Data", ["images", "yields", "locations", "indices", "years"]
        )

        train_data = Data(
            images=torch.as_tensor(
                train_images[:, :, :time, :], device=self.device
            ).float(),
            yields=torch.as_tensor(yields[train_idx], device=self.device)
            .float()
            .unsqueeze(1),
            locations=torch.as_tensor(locations[train_idx]),
            indices=torch.as_tensor(indices[train_idx]),
            years=torch.as_tensor(years[train_idx]),
        )

        test_data = Data(
            images=torch.as_tensor(
                test_images[:, :, :time, :], device=self.device
            ).float(),
            yields=torch.as_tensor(yields[test_idx], device=self.device)
            .float()
            .unsqueeze(1),
            locations=torch.as_tensor(locations[test_idx]),
            indices=torch.as_tensor(indices[test_idx]),
            years=torch.as_tensor(years[test_idx]),
        )

        return train_data, test_data

    @staticmethod
    def _normalize(train_images, val_images=None):
        """Normalize inputs using per-band means from the training split."""
        mean = np.mean(train_images, axis=(0, 2, 3))

        train_images = (train_images.transpose(0, 2, 3, 1) - mean).transpose(0, 3, 1, 2)
        val_images = (val_images.transpose(0, 2, 3, 1) - mean).transpose(0, 3, 1, 2)

        return train_images, val_images

    @staticmethod
    def analyze_results(true, pred):
        """Calculate ME, RMSE and MAE"""
        rmse = np.sqrt(np.mean((true - pred) ** 2))
        me = np.mean(true - pred)
        mae = np.mean(np.abs(true - pred))
        r_2 = 1 - np.sum((true - pred) ** 2) / \
            np.sum((true - np.mean(true)) ** 2)

        print(f"Without GP: RMSE: {rmse}, ME: {me}, MAE: {mae}, R_2: {r_2}")
        return rmse, me, mae, r_2

    def reinitialize_model(self, time=None):
        """Reset the model before starting a new training run."""
        raise NotImplementedError

    def _forward_model(self, inputs, years=None, return_last_dense=False):
        """Call the wrapped model with optional year inputs."""
        kwargs = {}
        if return_last_dense:
            kwargs["return_last_dense"] = True
        if getattr(self.model, "expects_year_input", False):
            kwargs["year"] = years
        return self.model(inputs, **kwargs)

    def evaluate_checkpoint(
        self,
        checkpoint_path,
        path_to_histogram,
        predict_year,
        time=32,
        batch_size=32,
    ):
        """Load a checkpoint and evaluate it on a target prediction year."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        with np.load(path_to_histogram) as hist:
            images = hist["output_image"]
            locations = hist["output_locations"]
            yields = hist["output_yield"]
            years = hist["output_year"]
            indices = hist["output_index"]

        train_data, test_data = self.prepare_arrays(
            images, yields, locations, indices, years, predict_year, time
        )
        results = self._predict(*train_data, *test_data, batch_size)
        rmse, me, mae, r_2 = self.analyze_results(
            results["test_real"],
            results["test_pred"],
        )
        return {
            "test_real": results["test_real"],
            "test_pred": results["test_pred"],
            "rmse": rmse,
            "me": me,
            "mae": mae,
            "r_2": r_2,
        }

    def test_gr(
        self,
        path_to_histogram,
        predict_year,
        time=32,
        batch_size=32,
    ):
        """Evaluate the current in-memory model without reloading a checkpoint."""
        with np.load(path_to_histogram) as hist:
            images = hist["output_image"]
            locations = hist["output_locations"]
            yields = hist["output_yield"]
            years = hist["output_year"]
            indices = hist["output_index"]

        train_data, test_data = self.prepare_arrays(
            images, yields, locations, indices, years, predict_year, time
        )
        results = self._predict(*train_data, *test_data, batch_size)
        rmse, me, mae, r_2 = self.analyze_results(
            results["test_real"],
            results["test_pred"],
        )
        return {
            "test_real": results["test_real"],
            "test_pred": results["test_pred"],
            "rmse": rmse,
            "me": me,
            "mae": mae,
            "r_2": r_2,
        }
