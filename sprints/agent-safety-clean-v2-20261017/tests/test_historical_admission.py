"""Historical reservation lineage fixtures; no global ledger writes."""
from copy import deepcopy
from pathlib import Path
import sys,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from historical_admission import validate_historical_admission


class HistoricalAdmissionTests(unittest.TestCase):
 def setUp(self):
  self.job={'job':'fixture','app_id':'ap-fixture','admitted_upper_bound_usd':20}
  self.reservation={'job':'fixture','admitted_upper_bound_usd':20,'upper_bound_usd':20}
 def call(self,history):
  with patch('launch_bounded_job.validate_resource_profile',return_value={'actual_source_check':'fixture'}) as profile:
   result=validate_historical_admission(self.job,self.reservation,history,ROOT)
   profile.assert_called_once_with({**self.job,'upper_bound_usd':20},ROOT)
  return result
 def test_later_reconciliation_preserves_original_profile_and_stable_audit(self):
  original=self.call([])
  self.reservation['upper_bound_usd']=3
  changes=[{'job':'fixture','app_id':'ap-fixture','previous_upper_bound_usd':20,'new_upper_bound_usd':8},
           {'job':'other','previous_upper_bound_usd':90,'new_upper_bound_usd':80},
           {'job':'fixture','app_id':'ap-fixture','previous_upper_bound_usd':8,'new_upper_bound_usd':3}]
  self.assertEqual(original,self.call(changes))
  self.assertEqual(self.reservation['upper_bound_usd'],3)
 def test_missing_disagreeing_or_invalid_original_bounds_fail(self):
  for value in (None,False,0,-1,float('nan'),float('inf'),19,'20'):
   self.job['admitted_upper_bound_usd']=value
   with self.subTest(value=value),self.assertRaises(ValueError):self.call([])
  del self.job['admitted_upper_bound_usd']
  with self.assertRaisesRegex(ValueError,'missing'):self.call([])
 def test_current_amount_requires_complete_connected_history(self):
  self.reservation['upper_bound_usd']=3
  cases=[[],[{'job':'fixture','previous_upper_bound_usd':8,'new_upper_bound_usd':3}],
   [{'job':'fixture','previous_upper_bound_usd':20,'new_upper_bound_usd':8}],
   [{'job':'fixture','previous_upper_bound_usd':20,'new_upper_bound_usd':21}],
   [{'job':'fixture','previous_upper_bound_usd':20,'new_upper_bound_usd':0}],
   [{'job':'fixture','previous_upper_bound_usd':20,'new_upper_bound_usd':3,'app_id':'ap-other'}],
   [{'job':'fixture','previous_upper_bound_usd':20,'new_upper_bound_usd':3}]*2]
  for history in cases:
   with self.subTest(history=history),self.assertRaises(ValueError):self.call(history)
 def test_reversed_history_is_not_sorted_into_acceptance(self):
  self.reservation['upper_bound_usd']=3
  history=[{'job':'fixture','previous_upper_bound_usd':8,'new_upper_bound_usd':3},
           {'job':'fixture','previous_upper_bound_usd':20,'new_upper_bound_usd':8}]
  with self.assertRaisesRegex(ValueError,'reordered'):self.call(history)
 def test_same_positive_bound_is_allowed_and_other_jobs_remain_separate(self):
  self.call([{'job':'fixture','previous_upper_bound_usd':20,'new_upper_bound_usd':20},
             {'job':'other','previous_upper_bound_usd':50,'new_upper_bound_usd':10}])
 def test_profile_failure_is_never_bypassed_by_good_reconciliation(self):
  with patch('launch_bounded_job.validate_resource_profile',side_effect=ValueError('source mismatch')):
   with self.assertRaisesRegex(ValueError,'source mismatch'):
    validate_historical_admission(self.job,self.reservation,[],ROOT)


if __name__=='__main__':unittest.main()
