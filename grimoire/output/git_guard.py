import git
from pathlib import Path
from grimoire.utils.logger import get_logger

logger = get_logger(__name__)

class GitGuard:
    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path).absolute()
        try:
            self.repo = git.Repo(self.vault_path, search_parent_directories=False)
        except git.InvalidGitRepositoryError:
            logger.warning("git_not_found", path=str(self.vault_path))
            self.repo = None

    def commit_pre_change(self, file_path: str, reason: str = "grimoire: pre-change snapshot"):
        if not self.repo:
            return

        try:
            # Check if there are changes in the specific file
            relative_path = Path(file_path).absolute().relative_to(self.repo.working_dir)
            
            if self.repo.is_dirty(path=relative_path):
                self.repo.index.add([str(relative_path)])
                self.repo.index.commit(f"{reason} for {relative_path}")
                logger.info("git_commit_success", file=file_path, reason=reason)
            else:
                logger.debug("git_no_changes_to_commit", file=file_path)
                
        except Exception as e:
            logger.error("git_commit_failed", file=file_path, error=str(e))

    def is_repo_ready(self) -> bool:
        return self.repo is not None
