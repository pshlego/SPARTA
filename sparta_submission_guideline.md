# SPARTA Submission Guidelines

Thank you for your interest in SPARTA (Scalable and Principled Benchmark of Tree-Structured Multi-Hop QA Over Text and Tables). This document provides detailed instructions for submitting your results to the SPARTA leaderboard.

---

## Overview

- **Dataset:** [https://huggingface.co/datasets/pshlego/SPARTA](https://huggingface.co/datasets/pshlego/SPARTA)
- **Contact:** pshlego@gmail.com / jekim@dblab.postech.ac.kr
- **Submission Frequency:** Each participant may submit **at most once per day** to the leaderboard.

The SPARTA dataset contains both **dev** and **test** workloads. Since the gold answers for the **dev set** have been publicly released, you do not need to submit dev set results to us. You can evaluate, analyze, and iterate on your method locally using the dev set. The leaderboard is exclusively for **test set** evaluation. If you want your results to be validated and displayed on the leaderboard, you must submit your test set predictions following the guidelines below.

---

## Submission Format

Your submission file must be in **JSON** format with the following structure:

```json
{
  "question_id_1": ["answer_1", "answer_2", ...],
  "question_id_2": ["answer_1"],
  ...
}
```

Each key is a **question ID** (string), and each value is a **list of answer strings**.

---

## Evaluation Tracks

SPARTA offers two evaluation tracks: **Oracle** and **Retrieval**. Each track has distinct requirements. Please read the relevant section carefully before submitting.

---

### Track 1: Oracle Setting

In the Oracle setting, your model is permitted to use the **table attributes** provided in the SPARTA test set available at [HuggingFace](https://huggingface.co/datasets/pshlego/SPARTA).

#### Required Submission Materials

| Material | Required | Description |
|---|---|---|
| **Prediction File** | Yes | A JSON file containing your model's predictions on the test set (format described above). |
| **Title** | Yes | The name of your method/model. |
| **Author** | Yes | Name(s) of the submitter(s). Enter `Anonymous` if you wish to remain anonymous. |
| **Description** | Yes | A clear description of your method, including the model architecture, key techniques, and any relevant details. |
| **Code Link** | Yes | A link to your code repository. See [Code Submission Requirements](#code-submission-requirements) below. |
| **Paper Link** | No | A link to a related paper, if available. |

---

### Track 2: Retrieval Setting

In the Retrieval setting, your model is **not permitted** to use the table attributes provided in the SPARTA test set. This track evaluates your model's ability to retrieve and reason over evidence without direct access to structured table metadata.

#### Required Submission Materials

| Material | Required | Description |
|---|---|---|
| **Prediction File** | Yes | A JSON file containing your model's predictions on the test set (format described above). |
| **Title** | Yes | The name of your method/model. |
| **Author** | Yes | Name(s) of the submitter(s). Enter `Anonymous` if you wish to remain anonymous. |
| **Description** | Yes | A clear description of your method. **Must include a detailed explanation of your retrieval strategy** (e.g., retrieval pipeline, indexing method, ranking approach). |
| **Code Link** | Yes | A link to your code repository. See [Code Submission Requirements](#code-submission-requirements) below. |
| **Paper Link** | No | A link to a related paper, if available. |

#### Additional Retrieval Evidence Requirements

To verify that your model did not use table attributes, you **must** submit a file named **`retrieved_evidence.json`** alongside your code:

```json
{
  "retrieved_tables": ["table_id_1", "table_id_2", ...],
  "retrieved_tuples": {"table_id": ["row_id_1", "row_id_2", ...]},
  "retrieved_passages": ["passage_id_1", "passage_id_2", ...]
}
```

- `retrieved_tables`: The source table IDs retrieved and used by your model.
- `retrieved_tuples`: A mapping from each table ID to the list of row IDs retrieved from that table.
- `retrieved_passages`: The passage IDs retrieved and used by your model.

This file must be included in your code submission to certify that your method operated without table attributes.

---

## Code Submission Requirements

For **both tracks**, you must provide access to your code. Please choose one of the following submission methods based on your preference regarding code visibility:

### Option A: Leaderboard Website — Code Link Will Be Public

If you are comfortable with your code being **publicly visible** on the leaderboard, submit directly through the leaderboard website with a code repository link.

1. Host your code on a platform such as GitHub, GitLab, or Google Drive.
2. **Grant access** to both `pshlego@gmail.com` and `jekim@dblab.postech.ac.kr` if the repository is private.
3. Your repository **must** include:
   - A **README** with clear instructions on how to reproduce your results.
   - **Execution logs** of your model's inference run on the test set.
   - (Retrieval track only) `retrieved_evidence.json` as described above.

> **Note:** The code link you provide will be displayed on the public leaderboard alongside your results.

### Option B: Email Submission — Code Remains Private

If you **do not** want your code to be publicly displayed on the leaderboard, first submit your prediction file through the leaderboard website as usual, then send a supplementary email to **both** **pshlego@gmail.com** and **jekim@dblab.postech.ac.kr** with your code. Your code will only be used for internal verification and will not be shared.

After submitting on the website, you will receive a unique **Submission ID** on the result page. You **must** include this **Submission ID** and the **date and time of your submission** in your email so we can match it to your leaderboard entry.

Please include the following in your email:
- **Submission ID** (displayed on the result page after your website submission)
- **Submission date and time** (e.g., `Feb 18, 2026, 12:53 AM`)
- Your code as a compressed archive (`.zip` or `.tar.gz`).
- A **README** with clear instructions on how to reproduce your results.
- Execution logs of your model's run.
- (Retrieval track only) `retrieved_evidence.json`.

### Execution Logs

All submissions **must** include execution logs demonstrating how your system solved the task. Logs should contain:
- Timestamps showing when each step was executed.
- The reasoning trace or intermediate outputs of your model.
- Any errors encountered during execution.

Logs can be in any text-based format (e.g., `.log`, `.txt`, `.json`, `.md`).

> **Why are logs required?** Execution logs allow us to verify that results were generated through a legitimate, automated pipeline. Adding timestamps to your logs strengthens the credibility of your submission.

---

## Email Submission Template

If you need to contact us regarding your submission (e.g., to provide supplementary materials or clarify details), please use the following email template:

**To:** pshlego@gmail.com, jekim@dblab.postech.ac.kr *(please send to both addresses)*

**Subject:** `[SPARTA Submission] #{Submission ID} - {Track: Oracle/Retrieval} - {Method Name}`

**Body:**

```
Dear SPARTA Team,

I would like to submit my results for the SPARTA leaderboard.

- Submission ID: [e.g., 42]
- Submission Date/Time: [e.g., Feb 18, 2026, 12:53 AM]
- Track: [Oracle / Retrieval]
- Method Name: [Your method name]
- Author(s): [Your name(s) or "Anonymous"]
- Description: [Brief description of your method]
- Code Link: [URL to your code repository, or state "attached"]
- Paper Link: [URL, if applicable]

[For Retrieval track only]
- Retrieval Strategy: [Description of how evidence was retrieved]
- Retrieved Evidence File: [Attached / included in code repository]

Attached:
1. Prediction file (JSON)
2. Execution logs
3. [Any additional files]

Best regards,
[Your name]
```

---

## Important Rules and Policies

### Fair Evaluation Policy

- You are welcome to conduct multiple experiments using various methods, frameworks, or models during development. However, **only one final prediction file is allowed per submission.**
- The process of determining the final result from multiple experiments must be an **autonomous decision** made by the model/agent itself (e.g., via majority voting or other predefined strategies). **Manually selecting the best result based on prior knowledge of the answers is strictly prohibited.**
> [!CAUTION]
> If you violate this rule, your result will **not** be allowed to appear on the leaderboard.

### Review Process and Timeline

- Your submission will **not** be visible on the leaderboard immediately. Results will become public only after our review is complete.
- **Expected review time:** Typically within **10 days** of submission. Depending on the complexity of your code and the clarity of your instructions, it may take up to **20 days**.
- If your submission is found to be **invalid**, we will notify you via email with the specific reason(s).
- If your submission has not become visible after **20 days** and you have not received any notification from us, please send a follow-up email to pshlego@gmail.com and jekim@dblab.postech.ac.kr.

> **Tip:** Providing a clear README with detailed reproduction instructions significantly speeds up the review process.

### Submission Frequency

- Each participant/team may submit **at most once per day** to the leaderboard.
- Please ensure your submission is thoroughly validated on the dev set before submitting to the test set.

### Data Protection and Integrity

- We will only use your code for evaluation and verification purposes.
- We will not disseminate or disclose any details of your code or methodology without your explicit consent.
- Please ensure your submission files are concise and contain only files relevant to your work.

### Disclaimer

- Submissions that are incomplete (e.g., missing required fields or execution logs) may be rejected or delayed.
- We reserve the right to request additional information or clarification about any submission.
- By submitting to the SPARTA leaderboard, you agree to abide by these guidelines.

---

## Quick Reference Checklist

### Oracle Track
- [ ] Prediction JSON file (correct format)
- [ ] Method title
- [ ] Author name(s)
- [ ] Method description
- [ ] Code with README (public link via website, or private via email)
- [ ] Execution logs included
- [ ] (Optional) Paper link

### Retrieval Track
- [ ] Prediction JSON file (correct format)
- [ ] Method title
- [ ] Author name(s)
- [ ] Method description (including retrieval strategy details)
- [ ] Code with README (public link via website, or private via email)
- [ ] Execution logs included
- [ ] `retrieved_evidence.json` (`retrieved_tables`, `retrieved_tuples`, `retrieved_passages`)
- [ ] (Optional) Paper link

---

## Contact

If you have any questions about the submission process, please reach out:

- **Email:** pshlego@gmail.com / jekim@dblab.postech.ac.kr
- **GitHub Issues:** [https://github.com/pshlego/SPARTA/issues](https://github.com/pshlego/SPARTA/issues)
- **Project Page:** [https://sparta-projectpage.github.io/](https://sparta-projectpage.github.io/)

We look forward to your submissions and thank you for contributing to the advancement of multi-hop question answering research!
