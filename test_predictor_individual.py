from utils.pat import IndividualPlotter
from predictor import Predictor


if __name__ == '__main__':
    building_index = 5
    dataset_type = 'price'

    expt_name = 'linear_L144_T48'
    predictor = Predictor(expt_name=expt_name, load=True)
    x, pred = predictor.test_individual(building_index, dataset_type)

    plotter = IndividualPlotter(x, pred, window_size=500)
    plotter.show()
