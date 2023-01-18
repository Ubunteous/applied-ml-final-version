from tools.landmarks import extract_features_labels
from tools.training import split_train_test, general_classifier, multi_train_test
from tools.plot import plot_data

from A1.Task_A1 import A1
from A2.Task_A2 import A2
from B1.Task_B1 import B1
from B2.Task_B2 import B2

for task_class in [A1, A2, B1, B2]:
    task = task_class.task
    label, max_data, data_dir, proportion_train = task_class().get_main_properties()

    print(f"Working on task {task} with label {label} from dataset {data_dir.name}\nProceeding to get {max_data} images including {proportion_train}% for training\n")

    X, y = extract_features_labels(data_dir, req_label=label, max_data=max_data)

    sorted_models, results = multi_train_test( *split_train_test(X, y, proportion_train = proportion_train) )

    plot_data(sorted_models, results, precision=task + ": "+ label, plot_bests_only=True, print_results=True, save=True)
