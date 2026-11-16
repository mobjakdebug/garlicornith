#!/usr/bin/env bash
# Put one run's logs in the branch (ci/last-run/<utc>-<job>/), so that a run can be read from anywhere: the
# artifact and log endpoints of Actions are not reachable from every environment.
#
#   ci/publish.sh <dir with the .txt files> <what this run was>
#
# It commits what is in <dir> and rebases onto the latest origin/<branch> on every attempt, so a run that publishes
# while the branch moves does not lose its logs to a stale push. Events pushed by GITHUB_TOKEN start no new run.
set -u
dir="$1"
what="${2:-run}"
branch="${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

git add -f "$dir" || exit 0
if git diff --cached --quiet; then
  echo "nothing to publish"
  exit 0
fi
git commit -q -m "ci: logs of $what" || exit 0

for attempt in 1 2 3 4 5; do
  if git push -q origin "HEAD:$branch"; then
    echo "published $dir"
    exit 0
  fi
  echo "push rejected (attempt $attempt), rebasing on origin/$branch"
  sleep 5
  git fetch -q origin "+refs/heads/$branch:refs/remotes/origin/$branch" || continue
  git rebase -q "origin/$branch" || { echo "rebase failed"; exit 0; }
done
echo "could not publish the logs"
exit 0
