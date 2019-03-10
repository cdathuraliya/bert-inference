import numpy as np
import tensorflow as tf
import bert.modeling as modeling
import bert.tokenization as tokenization
import bert.run_classifier as rc


class BertPrediction:

  BERT_CONFIG_FILE = "cased_L-12_H-768_A-12/bert_config.json"
  VOCAB_FILE = "cased_L-12_H-768_A-12/vocab.txt"

  do_lower_case = False
  max_seq_length = 128
  batch_size = 1
  is_training = False
  use_one_hot_embeddings = False
  bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG_FILE)
  tokenizer = tokenization.FullTokenizer(VOCAB_FILE, do_lower_case)

  gpu_config = tf.ConfigProto()
  gpu_config.gpu_options.allow_growth = True
  
  def __init__(self, processor, init_checkpoint):
    """Creates graphs, sessions and restore models.
    """
    self.label_list = processor.get_labels()
    self.graph = tf.Graph()

    with self.graph.as_default() as g:
      self.input_ids_p = tf.placeholder(tf.int32, [self.__class__.batch_size,
                                                   self.__class__.max_seq_length], name="input_ids")
      self.input_mask_p = tf.placeholder(tf.int32, [self.__class__.batch_size,
                                                    self.__class__.max_seq_length], name="input_mask")
      self.label_ids_p = tf.placeholder(tf.int32, [self.__class__.batch_size], name="label_ids")
      self.segment_ids_p = tf.placeholder(tf.int32, [self.__class__.max_seq_length], name="segment_ids")

      _, _, _, self.probabilities = rc.create_model(self.__class__.bert_config, self.__class__.is_training,
                                                    self.input_ids_p, self.input_mask_p, self.segment_ids_p,
                                                    self.label_ids_p, len(self.label_list),
                                                    self.__class__.use_one_hot_embeddings)
      saver = tf.train.Saver()
      graph_init_op = tf.global_variables_initializer()

    self.sess = tf.Session(graph=self.graph, config=self.__class__.gpu_config)
    self.sess.run(graph_init_op)
    
    with self.sess.as_default() as sess:
      saver.restore(sess, tf.train.latest_checkpoint(init_checkpoint))


  @staticmethod
  def convert_line(line, label_list, max_seq_length, tokenizer):
    """Function to convert a line that should be predicted into BERT
    input features.
    """
    label = tokenization.convert_to_unicode("0") # Mock label
    text_a = tokenization.convert_to_unicode(line)
    example = rc.InputExample(guid=0, text_a=text_a, text_b=None, label=label)
    feature = rc.convert_single_example(0, example, label_list, max_seq_length, tokenizer)

    input_ids = np.reshape([feature.input_ids], (1, max_seq_length))
    input_mask = np.reshape([feature.input_mask], (1, max_seq_length))
    segment_ids = np.reshape([feature.segment_ids], (max_seq_length))
    label_ids =[feature.label_id]

    return input_ids, input_mask, segment_ids, label_ids


  def run(self, line):
    """Function to run the inference
    """    
    input_ids, input_mask, segment_ids, label_ids = self.__class__.convert_line(line, self.label_list,
                                                                self.__class__.max_seq_length,
                                                                self.__class__.tokenizer)
    with self.graph.as_default() as g:
      with self.sess.graph.as_default():
        feed_dict = {self.input_ids_p: input_ids, self.input_mask_p: input_mask, 
                     self.segment_ids_p: segment_ids, self.label_ids_p: label_ids}
        prob = self.sess.run([self.probabilities], feed_dict)
        prob = prob[0][0] # get first label
        label_predict = np.argmax(prob)

        return label_predict, prob