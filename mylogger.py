import logging
import datetime
import time


class MultiMessageFormater(logging.Formatter):
    """
    기본적으로 log에는 하나의 message밖에 할당을 못함
    기존 formatter를 상속받아 추가적으로 message를 입력할 수 있도록 수정
    (**kwargs를 통해 임의의 입력에 대해 사용할 수는 없을까?)
    """
    def format(self, record):
        record.message2 = record.args.get("message2")
        return super().format(record)


class SaveLog:
    def __init__(self, log_name, metric):
        """
        작성할 log 종류, 모니터링할 metric의 종류, loss 및 metric을 입력받아 log 저장
        """
        super().__init__()

        self.log_name = "{}_{}".format(log_name, metric)
        self.loss = 0
        self.metric = 0
        self.logger = None
        self.handler = None

        self._init_handler()

    def _init_handler(self):
        self.logger = logging.getLogger(self.log_name)
        self.logger.setLevel(logging.DEBUG)

        self.handler = logging.FileHandler('./logfile_{:%Y%m%d}_{}.log'.format(datetime.datetime.now(), self.log_name), encoding='utf-8')
        formatter = MultiMessageFormater('%(asctime)s %(name)s loss: %(message)s metric: %(message2)s')
        self.handler.setFormatter(formatter)

    def save(self, val1, val2):
        self.logger.debug(val1, {"message2": val2})
        self.logger.addHandler(self.handler)


# logobj = SaveLog(log_name='monitoring train', metric='accuracy')