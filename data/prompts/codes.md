#### Reference Codes for Prompt Techniques

These were initially created for output filename tagging, but may be used elsewhere downstream.

```
adversarial-pleading:AP
anti-neutrality:AN
authority:A
extreme-persona:EP
foot-in-door:FID
moral-decoupling:MD
baseline:B
few-shot:FS
```

**Note on `few-shot`:** When `few-shot` is included in the prompts list, the system automatically loads opinion-specific examples from `fewshot_dir`. Each opinion (e.g., `A0`, `H5`) is matched with its corresponding few-shot examples (e.g., `A0_0`, `A0_1`, `A0_2`) from the CSV files. Three examples are inserted into the few-shot template per opinion. The few-shot prompt (including few-shot examples) is ALWAYS applied last.

#### Reference Codes for Topics

These are used to create index values in output files, and to match few-shot prompts to topics.

```
abortion:A
climate:B
crime:C
foreign-policy:D
gun-policy:E
healthcare:F
immigration:G
lgbtq-gender:H
speech:I
taxation:J
```

#### Few-Shot Example Naming Convention

Few-shot examples in CSV files use the format `{opinion_id}_{example_number}`:
- `A0_0`, `A0_1`, `A0_2` → Examples for opinion `A0` (abortion, position 0)
- `H5_0`, `H5_1`, `H5_2` → Examples for opinion `H5` (lgbtq-gender, position 5)