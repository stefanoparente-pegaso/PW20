class ModelResult:
    def __init__(self, name, accuracy, f1, conf_matrix):
        self.name = name
        self.accuracy = accuracy
        self.f1 = f1
        self.conf_matrix = conf_matrix

    def to_dict(self):
        return {
            "name": self.name,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "matrix": self.conf_matrix
        }