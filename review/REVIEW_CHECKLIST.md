# Review checklist

`team_verdict` is blank on purpose. `draft_verdict` was produced from the oracle's own reasoning, so copying it
is not a review. Check each row against the premise yourself and write what you find.

## Who does which rows
Row numbers count data rows (row 1 is the line after the header).

| Person | edge_review.csv | label_audit_review.csv | Disagreements re-checked by |
|---|---|---|---|
| Abhijeet | 1–8 | 1–2 | Prashant Pandey |
| Prashant Pandey | 9–16 | 3–4 | Divyanshi Khanka |
| Divyanshi Khanka | 17–24 | 5–6 | Shubham Jain |
| Shubham Jain | 25–31 | 7–8 | Rahul Kumar |
| Rahul Kumar | 32–38 | 9–10 | Abhijeet |

## Getting the premise
`label_audit_review.csv` has it in the `premise` column. For `edge_review.csv`, look it up by `doc_id`:
```bash
python -c "from c2c.data import load_splits; t = load_splits()['test']; print(t[t.doc_id == '0d42fe0540'].premise.iloc[0])"
```
Pasting the premise into the app's Corr2Cause tab draws the PC skeleton with its v-structures.

## edge_review.csv: is the edge cause → effect right?
Draw the skeleton: two variables are joined unless the premise says they are independent (given any set).

| Evidence says | What to check |
|---|---|
| `E is a collider of A and C: '<statement>' and E is not in that set` | A–E and C–E are joined, A–C are not, and E is missing from the conditioning set in the quoted statement. Then A → E ← C is forced. |
| `direction forced ... (propagated from the colliders)` | Reversing the edge would create a new collider or a cycle. |
| `adjacent but direction not identified: X->Y in only k of n` | The edge exists but some consistent DAG points it the other way, so claiming X → Y is not supported. |
| `not adjacent: the premise states '<statement>'` | The quoted statement is in the premise, so there is no edge at all. |

Values: `ok` (edge is entailed), `wrong` (not entailed or not adjacent), `unclear` (you cannot settle it; say why in a note to the double-checker).

## label_audit_review.csv: who is right, the dataset label or the oracle?
`label` is the dataset's answer, `oracle` is ours; they disagree on every row.

| Evidence says | What to check |
|---|---|
| `holds in n/n consistent DAGs ... so it is entailed` | Enumerate the DAGs that fit the premise. If the hypothesis holds in every one, the oracle is right. |
| `holds in 0/n under the strict definition, n/n under the looser one` | The answer depends on how the template is defined (e.g. whether "not a direct cause" also needs a longer path). Decide which definition the Corr2Cause paper uses. |

Values: `oracle` (oracle is right, label is wrong), `dataset` (label is right, oracle is wrong), `unclear` (depends on a definition nobody can pin down).

## Rules
- Fill `team_verdict` only with the values above. Do not edit any other column.
- If your verdict differs from `draft_verdict`, the person in the last column of the table re-checks that row.
  If you two still disagree, write `unclear`.
- Save as CSV with the same columns; keep the file's existing line endings.
