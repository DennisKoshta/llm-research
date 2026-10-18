"""Validate immutable original admission separately from later budget reductions.

Pure read-only helper: no ledger mutation, no inference and no legacy fallback.
The returned audit remains stable after legitimate later reconciliation, while
every invocation checks the complete current reduction chain independently.
"""
from decimal import Decimal
import math


def require(value,message):
 if not value:raise ValueError(message)


def amount(value):
 require(type(value) in (int,float) and math.isfinite(value) and value>0,'Admission amounts must be finite positive numbers')
 return Decimal(str(value))


def validate_historical_admission(job,reservation,reconciliations,root):
 from launch_bounded_job import validate_resource_profile
 require(isinstance(job,dict) and isinstance(reservation,dict) and isinstance(reconciliations,list),'Historical admission requires exact ledger rows and ordered history')
 require(job['job']==reservation['job'],'Historical reservation belongs to another job')
 require('admitted_upper_bound_usd' in job and 'admitted_upper_bound_usd' in reservation,'Original admitted bound missing; no legacy fallback')
 admitted=amount(job['admitted_upper_bound_usd'])
 require(admitted==amount(reservation['admitted_upper_bound_usd']),'Original admitted amounts disagree')
 prior=admitted
 for row in reconciliations:
  require(isinstance(row,dict) and isinstance(row.get('job'),str),'Malformed reconciliation record')
  if row['job']!=job['job']:continue
  require(amount(row['previous_upper_bound_usd'])==prior,'Reconciliation history is missing, reordered or disconnected')
  new=amount(row['new_upper_bound_usd'])
  require(new<=prior,'Reconciliation cannot increase the reserved bound')
  if 'app_id' in row:
   require(row['app_id']==job.get('app_id'),'Reconciliation belongs to another actual app')
  prior=new
 require(prior==amount(reservation['upper_bound_usd']),'Current reservation differs from complete reconciliation history')
 checked=validate_resource_profile({**job,'upper_bound_usd':job['admitted_upper_bound_usd']},root)
 return {'schema':'historical_resource_admission_audit_v1','job':job['job'],
  'admitted_upper_bound_usd':job['admitted_upper_bound_usd'],'resource_profile_audit':checked,
  'reconciliation_chain_verified':True}
