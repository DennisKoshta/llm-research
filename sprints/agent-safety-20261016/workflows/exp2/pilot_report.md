# Exp2 pilot result

The repaired source task produced the intended role contrast: all 8 instructed colluders selected source Option B, while all 24 honest agent decisions selected Option A. All 32 repaired votes were valid. This is a feasibility check on four source scenarios, not an estimate of heldout probe performance.

The fresh permission task yielded 64/64 valid decisions. Authorized contributors submitted 16/16 times; unauthorized contributors submitted 0/16 times. Authorization competence passed, but actual violation prediction may be unidentifiable if the main task also has no violations.

The original source final-vote format failed 0/32 strict valid responses: outputs copied the option text instead of returning a single letter. Those raw failures and the original frozen source remain preserved. An approved named v2 repair appended a format instruction to the final system/user messages and reran only the 32 final decisions from preserved histories. No role, task option, reward or substantive instruction changed. All 168 collected calls have finite prospective captures at layers 26, 28 and 30.

The source probe analysis is a BF16, short-discussion, prospective-input adaptation of NARCBench. Main training/calibration/test splits hold out entire domains and entire conversations. Fresh specificity splits hold out complete task families; all agents and authorization variants remain together. Main inference requires the root approval gate after the analysis implementation is frozen.
