import ReadCsv
import unittest



class TestAudioTriggerRecorder(unittest.TestCase):
   def setUp(self):
       self.audio_trigger = AudioTrigger()

   def tearDown(self):
       pass

   def test_recorder(self):
       input_stream = InputStreamFactory().stream
       self.assertIsInstance(input_stream, pyaudio.Stream)
       current_data = input_stream.read(audio_config.CHUNK)
       self.assertTrue(len(current_data))
       self.assertIsInstance(current_data, str)
       boolean_threshold = self.audio_trigger.process(current_data)
       self.assertIsInstance(boolean_threshold, bool)
       input_stream.close()
       self.ass