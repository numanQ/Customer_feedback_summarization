"""Script for testing functions in inference.py"""

import pytest
from testing.mock_predict import mock_output
#pylint: disable=unused-argument

def mock_predict(file_path):
  """Mock function which return mock Predictor class with predict method"""

  class MockPredictor:

    def predict(self, request_body):
      """when called returns mocked output of get_prediction method"""
      return mock_output
  return MockPredictor()


@pytest.mark.parametrize(
    "request_body, mock_const_parse_tree",
    [({"sentence": "The dog is running."},
      "(S (NP (DT The) (NN dog)) (VP (VBZ is) (VP (VBG running))) (. .))")
    ])
def test_get_prediction(mocker, request_body, mock_const_parse_tree):
  mocker.patch(
      "allennlp.predictors.predictor.Predictor.from_path",
      side_effect=mock_predict)
  # pylint: disable=C0415
  from services.inference import get_prediction
  result = get_prediction(request_body=request_body)
  assert "trees" in result
  assert result["trees"] == mock_const_parse_tree
