from Logistic.dataprocess import Data_process



DATA_ROOT = "../Data/test_data/"
class Model_test(object):
    def __init__(self, task_id, model_id, model_dict, filename):
        self.test_model_dict = model_dict
        self.task_id = task_id
        self.filename = filename
        self.model_id = model_id
        # self.model_name = \
        #     {
        #         "1":"word2vec"
        #     }
        
        # embedding_word2vec
        self.model_load_pre = self.test_model_dict[str(task_id)][0] + '_' + self.test_model_dict[str(task_id)][1][str(self.model_id)]
        self.model_load = self.model_load_pre + '_evaluation'

    def process(self):
        self.load_data()
        self.model_test()



    def load_data(self):
        print("Start loading data...")
        self.text = Data_process.read_file(DATA_ROOT + self.filename)


    def model_test(self):
        print("Start testing model...")
        exec('from Logistic.evaluation import ' + self.model_load)
        exec(self.model_load + ".test(self.text)")

# task = Model_test(1,1)
# task.process()




