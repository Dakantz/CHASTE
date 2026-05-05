# llama-cpp-beam-search
Creating Moodle XML Questions from extended GIFT format

## Features

* Parse GIFT files more robustly compared to moodle
* Supports full markdown syntax (incl. Code highlighting!)
* Points are automatically inferred OR can be set!
  - Deducts -50% for single choice, -100% for T/F question
* Supports:
 - True/False Question (will be converted to Multichoice to allow point deduction)
 - Multichoice
 - Single Choice
 - Fill-the Blank



## Install

```
pip install gift-to-moodlexml
```

## Usage

Assume you have some (extended) GIFT file:

```md
$CATEGORY: OOP2/Intro

[markdown]Java supports use of `varargs` (variable arguments) for parameter passing {T}

[markdown] What pattern does this Code use?
`Logger logger=Logger.getInstance();`
{
    ~ Factory
    ~ Configurator
    = Singleton
    ~ Generics
    ~ Builder
    ~ Creator
    ~ Observer
}
```
Then you can parse it to an XML to upload to moodle:

```python
import gift_to_moodlexml
from pathlib import Path
questions_files= list(Path("./").glob("*.gift"))
all_questions = []
for q_file in questions_files:
    with open(q_file, "r") as file:
        content = file.read()
        questions = content.split("\n\n")
        questions = [q for q in questions]
        question = questions[0]
        all_questions.extend(questions)
gift_to_moodlexml.generate_xml_from_questions(all_questions, output_file="quiz_package.xml")
```