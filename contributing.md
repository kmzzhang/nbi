# Contributing

`nbi` is released under the BSD license. We encourage you to modify it, reuse it, and contribute changes back for the
benefit of others. We follow standard open source development practices: changes are submitted as pull requests and,
once they pass the test suite, reviewed by the team before inclusion.

To contribute back via pull requests, please first fork this report and make sure that you add the original repo as
the `upstream` remote:

```bash
git remote add upstream https://github.com/kmzzhang/nbi.git
```

You'll need to add `pre-commit` hooks by:

```bash
pip install pre-commit
pre-commit install
```

This will only need to be done once. After that, all your commits will be checked for code formatting issues.

Make a local branch like:

```bash
git switch -c your_local_branch_with_changes_name
```

Then commit to this branch as normal. When you are ready to submit a PR push to your fork:

```bash
git push -u origin <your_local_branch_with_changes_name>
```

Then create a PR back to `upstream`:

```
https://github.com/<your_username>/nbi/pull/new/<your_local_branch_with_changes_name>
```
