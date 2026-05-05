# GenAI-Assisted-Labelling

**Generative AI assisted labelling of GitHub issues and pull requests.**

## Dependencies

 * Python interpreter
 * GitHub command-line client, `gh`: https://cli.github.com
 * Access to a generative AI service which the script can prompt
   - ChatGPT subscription with Codex (https://chatgpt.com/codex) access, and the `codex` binary set up and logged in to

## Usage

Navigate to the Git checkout working directory of the project you want to label, and run the script:

```bash
$ <path to this project>/ai-labelling
```

Or run it with `--repository <owner>/<repository>` explicitly:

```bash
$ ./ai-labelling --repository <owner>/<repository>
```

By default, open issues updated in the last 24 hours are checked.
Additional options are available to match closed issues, PRs open and closed, change the time frame, select by creation date instead of update date, and more.

**See _`-h`_ for options' documentation.**
