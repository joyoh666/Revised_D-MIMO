from runtime.csi_producer import csi_producer
from runtime.csi_consumer import csi_consumer
from runtime.logger import result_logger

from multiprocessing import Queue, Process

def main():
    csi_queue = Queue(maxsize=1000)
    result_queue = Queue()
    block_queue = Queue()

    producer = Process(target=csi_producer, args=(csi_queue,block_queue))
    consumer = Process(target=csi_consumer, args=(csi_queue, result_queue))
    logger = Process(target=result_logger, args=(result_queue,))

    producer.start()
    consumer.start()
    logger.start()

    producer.join()
    consumer.join()
    logger.join()

if __name__ == "__main__":
    main()