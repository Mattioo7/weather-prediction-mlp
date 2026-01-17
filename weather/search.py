from dataclasses import replace
from itertools import product

from weather.dataset import build_dataset
from weather.normalization import normalize_global
from mlp.mlp import MLP

from weather.config import (
    Experiment,
    WeatherGridParams,
    MLPGridParams,
    FitGridParams,
)

def _log(msg: str):
    print(f"{msg}")


def expand_grid(params):
    keys = params.__dataclass_fields__.keys()
    axes = []

    for key in keys:
        value = getattr(params, key)

        # ONLY list means grid dimension
        if isinstance(value, list):
            axes.append(value)
        else:
            axes.append([value])

    for combo in product(*axes):
        yield replace(params, **dict(zip(keys, combo)))


class Search:
    """
    Minimal orchestration:
    - build_dataset()
    - train_model()
    - run()
    """

    # ======================================================
    # 1. BUILD DATASET
    # ======================================================
    def build_dataset(
        self,
        experiment: Experiment,
        weather_grid: WeatherGridParams,
    ):
        _log(f"\nBuilding dataset")

        # --- TRAIN ---
        _log("  → TRAIN split")
        wf = experiment.weather_fixed
        wg = weather_grid

        cfg_train = self._to_weather_config(wf, wg, split="train")
        X_train, y_train = build_dataset(cfg_train)
        X_train, mu, sigma = normalize_global(X_train)

        # --- TEST ---
        _log("  → TEST split")
        cfg_test = self._to_weather_config(wf, wg, split="test")
        X_test, y_test = build_dataset(cfg_test)
        X_test = (X_test - mu) / sigma

        return X_train, y_train, X_test, y_test

    # ======================================================
    # 2. CREATE + TRAIN MODEL
    # ======================================================
    def train_model(
        self,
        experiment: Experiment,
        mlp_grid: MLPGridParams,
        fit_grid: FitGridParams,
        X_train,
        y_train,
    ) -> tuple[MLP, list[float], list[list[float]], list[float]]:
        mf = experiment.mlp_fixed
        mg = mlp_grid
        ff = experiment.fit_fixed
        fg = fit_grid

        _log(
            f"\nTraining model"
        )

        model = MLP(
            layer_sizes=[X_train.shape[1], *mg.hidden_layers, 1],
            task=mf.task,
            activation=mg.activation,
            learning_rate=mg.learning_rate,
            seed=mg.seed,
            use_bias=mg.use_bias,
            loss=mg.loss,
            optimizer=mg.optimizer,
            beta=mf.beta,
            beta2=mf.beta2,
            eps=mf.eps,
            adaptive_lr=mf.adaptive_lr,
            lr_decay=mf.lr_decay,
        )

        history, weight_history, accuracy_history = model.fit(
            X_train,
            y_train,
            epochs=fg.epochs,
            batch_size=fg.batch_size,
            shuffle=fg.shuffle,
            verbose=ff.verbose,
            log_every=ff.log_every,
            use_tqdm=ff.use_tqdm,
            one_hot_if_needed=ff.one_hot_if_needed,
            early_stopping=ff.early_stopping,
            patience=ff.patience,
            min_delta=ff.min_delta,
            val_split=fg.val_split,
        )

        return model, history, weight_history, accuracy_history

    # ======================================================
    # 3. BUILD DATASET + TRAIN (FULL RUN)
    # ======================================================
    def run(self, experiment: Experiment):
        _log(f"\nStarting experiment: {experiment.name}")

        results = []
        run_id = 0

        for wg in expand_grid(experiment.weather_grid):
            X_train, y_train, X_test, y_test = self.build_dataset(experiment, wg)

            for mg in expand_grid(experiment.mlp_grid):
                for fg in expand_grid(experiment.fit_grid):
                    run_id += 1

                    model, history, weight_history, accuracy_history = self.train_model(
                        experiment,
                        mg,
                        fg,
                        X_train,
                        y_train,
                    )

                    # TODO: Return results or model?
                    y_pred = model.predict(X_test)

                    # probabilities only for classification
                    y_proba = None
                    if model.task in ("binary", "multiclass"):
                        y_proba = model.predict_proba(X_test)

                    results.append({
                        "experiment": experiment.name,
                        "weather": wg,
                        "mlp": mg,
                        "fit": fg,

                        "model": model,

                        # training artifacts
                        "history": history,
                        "weight_history": weight_history,
                        "accuracy_history": accuracy_history,

                        # test outputs
                        "y_test": y_test,
                        "y_pred": y_pred,
                        "y_proba": y_proba,
                    })

        _log(f"\nExperiment finished | total runs = {run_id}\n")
        return results

    def _to_weather_config(self, fixed, grid, split):
        from weather.config_weather import WeatherConfig

        cfg = WeatherConfig(
            data_dir=fixed.data_dir,
            split=split,
            target=fixed.target,
            target_mode=fixed.target_mode,
            target_threshold=fixed.target_threshold,
            skip_day=fixed.skip_day,
            hours_per_day=fixed.hours_per_day,
            normalization=fixed.normalization,
            encode_wind_direction=fixed.encode_wind_direction,
            include_city_coords=fixed.include_city_coords,
            max_missing_ratio_per_day=fixed.max_missing_ratio_per_day,
        )

        cfg.window_aggregation = grid.window_aggregation
        cfg.window_size = grid.window_size
        cfg.input_variables = grid.input_variables
        cfg.aggregations = grid.aggregations
        cfg.cities = grid.cities

        return cfg
