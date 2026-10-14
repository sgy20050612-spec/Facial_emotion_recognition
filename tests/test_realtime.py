import queue
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from realtime_fer.engine import Smoother, preprocess, softmax
from realtime_fer.app import capture_worker, open_capture


class RealtimeTests(unittest.TestCase):
    def test_camera_auto_skips_unreadable_device(self):
        class Capture:
            def __init__(self, good): self.good, self.released = good, False
            def isOpened(self): return True
            def set(self, *args): pass
            def read(self): return (True, np.zeros((2,2,3), np.uint8)) if self.good else (False, None)
            def release(self): self.released = True
        captures = [Capture(False), Capture(False), Capture(True)]
        with patch('realtime_fer.app.cv2.VideoCapture', side_effect=captures) as factory:
            result = open_capture('自动', True)
        self.assertIs(result, captures[2])
        self.assertTrue(all(c.released for c in captures[:2]))
        self.assertFalse(result.released)
        self.assertEqual([c.args[0] for c in factory.call_args_list], [0, 0, 1])

    def test_manual_camera_does_not_switch_device(self):
        from unittest.mock import MagicMock
        cap = MagicMock()
        cap.isOpened.return_value = False
        with patch('realtime_fer.app.cv2.VideoCapture', return_value=cap) as factory:
            with self.assertRaises(RuntimeError):
                open_capture('2', True)
        self.assertEqual([c.args[0] for c in factory.call_args_list], [2, 2])
        self.assertEqual(cap.release.call_count, 2)

    def test_preprocessing_rgb_order_and_normalization(self):
        bgr = np.zeros((12, 15, 3), np.uint8)
        bgr[:, :, 2] = 255
        x = preprocess(bgr)
        self.assertEqual(x.shape, (1, 3, 224, 224))
        np.testing.assert_allclose(x[0, :, 0, 0], [(1-.485)/.229, -.456/.224, -.406/.225], rtol=1e-5)

    def test_softmax_stability(self):
        p = softmax(np.arange(8) + 10000)
        self.assertAlmostEqual(float(p.sum()), 1., places=6)
        self.assertEqual(p.argmax(), 7)
        with self.assertRaises(ValueError):
            softmax(np.zeros(7))

    def test_no_stale_faces_and_no_cross_person_smoothing(self):
        s = Smoother()
        a, b = np.eye(8)[0], np.eye(8)[4]
        first = s.update([((0,0,40,40), a), ((100,0,140,40), b)], 0.)
        second = s.update([((102,0,142,40), b), ((2,0,42,40), a)], .1)
        self.assertEqual(first[0]['track_id'], second[1]['track_id'])
        self.assertEqual(first[1]['track_id'], second[0]['track_id'])
        self.assertEqual(s.update([], .2), [])
        after = s.update([((0,0,40,40), b)], 2.)
        self.assertNotEqual(after[0]['track_id'], first[0]['track_id'])
        np.testing.assert_allclose(after[0]['probabilities'], b)

    def test_invalid_source_reports_error_and_releases(self):
        class FakeCapture:
            released = False
            def isOpened(self): return False
            def release(self): self.released = True
        cap = FakeCapture()
        events = queue.Queue()
        with self.assertLogs(level='ERROR'), patch('realtime_fer.app.Recognizer'), patch('realtime_fer.app.open_capture', return_value=cap):
            capture_worker('missing.mp4', False, False, Path('.'), threading.Event(), threading.Event(), queue.Queue(1), events)
        self.assertTrue(cap.released)
        kinds = [item[0] for item in list(events.queue)]
        self.assertIn('error', kinds)
        self.assertEqual(kinds[-1], 'done')

    def test_video_eof_full_processing(self):
        class FakeEngine:
            def process(self, frame, timestamp): return []
        temp = Path('realtime_output/unit_tests')
        temp.mkdir(parents=True, exist_ok=True)
        video = str(temp / 'test.avi')
        writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*'MJPG'), 60, (64,64))
        self.assertTrue(writer.isOpened())
        for i in range(4): writer.write(np.full((64,64,3), i*30, np.uint8))
        writer.release()
        events, frames = queue.Queue(), queue.Queue(1)
        with patch('realtime_fer.app.Recognizer', return_value=FakeEngine()):
            capture_worker(video, False, False, Path('.'), threading.Event(), threading.Event(), frames, events)
        self.assertEqual(frames.get()['index'], 4)
        self.assertNotIn('error', [item[0] for item in list(events.queue)])


if __name__ == '__main__':
    unittest.main()
