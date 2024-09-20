import numpy as np
import pandas as pd
import tensorflow as tf
from stable_baselines3.common.callbacks import BaseCallback
import os
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results

#############################################
#### HILFSFUNKTIONEN FÜR DIE MUSTERGENERIERUNG
#############################################

def create_training_data(data, input_col, target_col, window_size=1, training_pattern_percent=0.7):
    """
    Erzeugt Trainings- und Validierungsdaten für das Modell.
    
    :param data: Das Daten-DataFrame.
    :param input_col: Liste der Eingabespaltennamen.
    :param target_col: Liste der Zielspaltennamen.
    :param window_size: Größe des historischen Fensters.
    :param training_pattern_percent: Prozentsatz der Daten, der für das Training verwendet wird.
    :return: Tuple mit Trainings- und Validierungsdaten sowie Normalisierungsparametern.
    """
    data_train = data

    mean_in, std_in = mean_and_std(input_col, data_train)
    mean_out, std_out = mean_and_std(target_col, data_train)

    print(f"Durchschnitt Eingaben = {mean_in}")
    print(f"Standardabweichung Eingaben = {std_in}")
    print(f"Durchschnitt Zielwerte = {mean_out}")
    print(f"Standardabweichung Zielwerte = {std_out}")

    grouped = data_train.groupby(['episode'])

    inputs_all = []
    labels_all = []

    for g in grouped:
        # Stelle sicher, dass die Daten innerhalb einer Gruppe nicht gemischt werden
        g = g[1].sort_values(by='step')

        past_history = window_size   # t-3, t-2, t-1, t
        future_target = 0  # t+1
        STEP = 1  # Keine Unterabtastung der Zeilen

        # Verwende pandas.DataFrame.values, um ein numpy-Array aus einem pandas.DataFrame-Objekt zu erhalten
        inputs, labels = multivariate_data(
            dataset=g[input_col][:].values,
            target=g[target_col][:].values,
            start_index=0,
            end_index=g[input_col][:].values.shape[0] - future_target,
            history_size=past_history,
            target_size=future_target,
            step=STEP,
            single_step=True
        )

        # Füge Daten zum gesamten Satz von Mustern hinzu
        for i in range(len(inputs)):
            inputs_all.append(inputs[i])
            labels_all.append(labels[i])

    length = len(inputs_all)

    # Mische die Daten
    c = list(zip(inputs_all, labels_all))
    np.random.shuffle(c)
    inputs_all, labels_all = zip(*c)

    split = int(training_pattern_percent * length)

    inputs_all = np.array(inputs_all)
    labels_all = np.array(labels_all)

    return ((inputs_all[:split], labels_all[:split]), (inputs_all[split:], labels_all[split:])), mean_in, std_in, mean_out, std_out


def mean_and_std(columns, data):
    """
    Berechnet den Durchschnitt und die Standardabweichung für gegebene Spalten.

    :param columns: Liste der Spaltennamen.
    :param data: Das Daten-DataFrame.
    :return: Durchschnitt und Standardabweichung für jede Spalte.
    """
    mean = np.zeros(len(columns))
    std = np.zeros(len(columns))
    index = 0
    for c in columns:
        mean[index], std[index] = get_normalizations(data[c])
        index += 1
    return mean, std


def get_normalizations(data):
    """
    Berechnet den Durchschnitt und die Standardabweichung eines Datensatzes.

    :param data: Die Datenreihe.
    :return: Durchschnitt und Standardabweichung.
    """
    mean = data.mean()
    std = data.std()
    return mean, std


def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=False):
    """
    Erstellt multivariate Daten für Zeitreihenanalysen.

    :param dataset: Das Eingabedaten-Array.
    :param target: Das Zielwerte-Array.
    :param start_index: Startindex für das Fenster.
    :param end_index: Endindex für das Fenster.
    :param history_size: Größe des historischen Fensters.
    :param target_size: Größe des Zielfensters.
    :param step: Schrittweite für die Fenstererstellung.
    :param single_step: Wenn True, wird nur der nächste Zeitstempel als Ziel verwendet.
    :return: Arrays für Eingaben und Zielwerte.
    """
    data = []
    labels = []

    start_index += history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])

    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.float32)


def prepare_data(df, input_col, target_col, window_size, training_batch_size=50, validation_batch_size=50, training_pattern_percent=0.7):
    """
    Bereitet die Trainings- und Validierungsdaten für das Modell vor.

    :param df: Das Daten-DataFrame.
    :param input_col: Liste der Eingabespaltennamen.
    :param target_col: Liste der Zielspaltennamen.
    :param window_size: Größe des historischen Fensters.
    :param training_batch_size: Batch-Größe für das Training.
    :param validation_batch_size: Batch-Größe für die Validierung.
    :param training_pattern_percent: Prozentsatz der Daten, der für das Training verwendet wird.
    :return: Train- und Validierungsdaten, Eingabeform und Normalisierungsparameter.
    """
    global x_train_multi, y_train_multi

    ###################
    ## DATEN VORBEREITEN
    ###################
    ((x_train_multi, y_train_multi), (x_val_multi, y_val_multi)), mean_in, std_in, mean_out, std_out = \
        create_training_data(
            df, input_col, target_col, window_size=window_size,
            training_pattern_percent=training_pattern_percent
        )

    print('Trainingsdaten: Einzelnes Fenster der Vergangenheit : {}'.format(x_train_multi[0].shape))
    print('Trainingsdaten: Einzelnes Fenster der Zukunft : {}'.format(y_train_multi[1].shape))
    print('Validierungsdaten: Einzelnes Fenster der Vergangenheit : {}'.format(x_val_multi[0].shape))
    print('Validierungsdaten: Einzelnes Fenster der Zukunft : {}'.format(y_val_multi[1].shape))
    print('Trainingsdaten: Anzahl der Trainingsbeispiele: {}'.format(x_train_multi.shape))
    print('Validierungsdaten: Anzahl der Validierungsbeispiele: {}'.format(x_val_multi.shape))

    train_data = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
    train_data = train_data.shuffle(x_train_multi.shape[0]).batch(training_batch_size).repeat()

    val_data = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
    val_data = val_data.batch(validation_batch_size).repeat()
    
    input_shape = x_train_multi[0].shape[-2:]
    
    return train_data, val_data, input_shape, mean_in, std_in, mean_out, std_out


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param avg_episodes: (int) average over the past n episodes (default: 100)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    :param vec_norm_env: (VecNormalize) stable-baselines VecNormalize object (contains Gym env)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1, avg_episodes=100, vec_norm_env=None):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.best_timestep = -np.inf
        self.vec_norm_env = vec_norm_env
        self.avg_episodes = avg_episodes

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            
            if len(x) > 0:
                # Mean training reward over the last avg_episodes episodes
                mean_reward = np.mean(y[-self.avg_episodes:])

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.best_timestep = self.num_timesteps
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)
                    if self.vec_norm_env is not None: 
                        self.vec_norm_env.save ("%s.env_normalizations" % self.save_path)

                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("Best mean reward: {:.2f} (ts={}) - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, self.best_timestep, mean_reward))


        return True