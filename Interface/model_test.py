from Logistic.dataprocess import Data_process
from Logistic.evaluation import word2vec_evaluation


DATA_ROOT = "../Data/test_data/"
class Model_test(object):
    def __init__(self, task_id, model_id, filename):
        self.task_id = task_id
        self.filename = filename
        self.modelid = model_id
        self.model_name = \
            {
                "1":"word2vec"
            }

    def process(self):
        # 嵌入任务
        if self.task_id == 1:
            self.load_data()
            self.model_test()
        elif self.task_id == 2:
            self.load_data()


    def load_data(self):
        print("Start loading data...")
        self.text = Data_process.read_file(DATA_ROOT + self.filename)


    def model_test(self):
        print("Start testing model...")
        word2vec_evaluation.test(self.text)

# task = Model_test(1,1)
# task.process()




