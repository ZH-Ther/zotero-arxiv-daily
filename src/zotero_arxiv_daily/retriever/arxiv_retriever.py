from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
import feedparser
from tqdm import tqdm
import multiprocessing
import os
from queue import Empty
from time import sleep
from urllib.request import urlopen
from typing import Any, Callable, TypeVar
from loguru import logger
import requests

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str, paper_title: str | None = None) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")
    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        """
        Retrieve latest arXiv papers from subscribed categories.
    
        Main changes:
        1. Fetch each arXiv category RSS feed separately instead of joining categories with '+'.
        2. Use feed.feed.get("title", "") to avoid AttributeError when RSS parsing fails.
        3. Add request intervals to reduce arXiv 429/503 rate-limit errors.
        4. Deduplicate paper IDs across categories.
        5. Retry arXiv API batches with backoff for 429/503.
        """
    
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
    
        categories = self.config.source.arxiv.category
        if categories is None:
            raise ValueError("category must be specified for arxiv.")
    
        # Be tolerant if a single category is accidentally provided as a string.
        if isinstance(categories, str):
            categories = [categories]
    
        raw_papers = []
        all_paper_ids = []
        seen_ids = set()
    
        # arXiv asks legacy API users not to exceed one request every 3 seconds.
        # Use a larger value to be safer on GitHub Actions shared runners.
        RSS_REQUEST_INTERVAL = 6
        API_REQUEST_INTERVAL = 12
        RSS_TIMEOUT = 30
    
        max_rss_retries = 3
        rss_retry_delay = 30
    
        # ------------------------------------------------------------------
        # Step 1: Retrieve RSS feeds category by category.
        # ------------------------------------------------------------------
        for category_index, category in enumerate(categories):
            if category_index > 0:
                sleep(RSS_REQUEST_INTERVAL)
    
            feed_url = f"https://rss.arxiv.org/atom/{category}"
            logger.info(f"Fetching arXiv RSS: {feed_url}")
    
            feed = None
    
            for attempt in range(max_rss_retries):
                try:
                    with urlopen(feed_url, timeout=RSS_TIMEOUT) as response:
                        feed = feedparser.parse(response)
                    break
                except Exception as exc:
                    if attempt < max_rss_retries - 1:
                        wait = rss_retry_delay * (attempt + 1)
                        logger.warning(
                            f"Failed to fetch arXiv RSS {feed_url}. "
                            f"Retry {attempt + 1}/{max_rss_retries} in {wait}s. "
                            f"Reason: {type(exc).__name__}: {exc}"
                        )
                        sleep(wait)
                    else:
                        raise Exception(
                            f"Failed to fetch arXiv RSS {feed_url} after "
                            f"{max_rss_retries} attempts. "
                            f"Reason: {type(exc).__name__}: {exc}"
                        )
    
            if feed is None:
                continue
    
            logger.info(f"Fetched arXiv RSS: {feed_url}, entries={len(feed.entries)}")
    
            # Avoid AttributeError when feed.feed has no title field.
            feed_title = feed.feed.get("title", "")
    
            if "Feed error for query" in feed_title:
                raise Exception(f"Invalid ARXIV_QUERY: {category}.")
    
            # feedparser may set bozo=True for malformed feeds.
            # If entries still exist, continue with a warning.
            # If no entries exist, skip this category instead of crashing the whole workflow.
            if getattr(feed, "bozo", False):
                logger.warning(
                    f"arXiv RSS feed may be malformed: {feed_url}. "
                    f"Reason: {getattr(feed, 'bozo_exception', 'unknown')}"
                )
                if len(feed.entries) == 0:
                    logger.warning(f"Skip empty malformed RSS feed: {feed_url}")
                    continue
    
            for item in feed.entries:
                announce_type = item.get("arxiv_announce_type", "new")
                if announce_type not in allowed_announce_types:
                    continue
    
                paper_id = item.id.removeprefix("oai:arXiv.org:")
    
                if paper_id not in seen_ids:
                    seen_ids.add(paper_id)
                    all_paper_ids.append(paper_id)
    
        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]
    
        logger.info(f"Collected {len(all_paper_ids)} unique arXiv paper ids")
    
        if len(all_paper_ids) == 0:
            logger.warning("No arXiv papers collected from RSS feeds.")
            return raw_papers
    
        # ------------------------------------------------------------------
        # Step 2: Retrieve full paper metadata from arXiv API.
        # ------------------------------------------------------------------
        # Smaller page_size and larger delay are more stable under GitHub Actions.
        client = arxiv.Client(
            page_size=5,
            delay_seconds=API_REQUEST_INTERVAL,
            num_retries=3,
        )
    
        bar = tqdm(total=len(all_paper_ids))
    
        api_batch_size = 5
        max_batch_retries = 6
        batch_retry_base_delay = 60
    
        # Wait before switching from RSS requests to arXiv API requests.
        sleep(API_REQUEST_INTERVAL)
    
        for i in range(0, len(all_paper_ids), api_batch_size):
            paper_ids = all_paper_ids[i:i + api_batch_size]
    
            logger.info(
                f"Fetching arXiv API batch {i // api_batch_size + 1}: "
                f"{len(paper_ids)} papers"
            )
    
            search = arxiv.Search(
                id_list=paper_ids,
                max_results=len(paper_ids),
            )
    
            for attempt in range(max_batch_retries):
                try:
                    batch = list(client.results(search))
                    raw_papers.extend(batch)
                    bar.update(len(paper_ids))
                    break
    
                except arxiv.HTTPError as exc:
                    status = getattr(exc, "status", None)
    
                    if status in (429, 503) and attempt < max_batch_retries - 1:
                        wait = batch_retry_base_delay * (attempt + 1)
    
                        logger.warning(
                            f"arXiv API HTTP {status} on batch "
                            f"{i // api_batch_size + 1}. "
                            f"Retry {attempt + 1}/{max_batch_retries} in {wait}s."
                        )
    
                        sleep(wait)
                        continue
    
                    raise
    
            if i + api_batch_size < len(all_paper_ids):
                sleep(API_REQUEST_INTERVAL)
    
        bar.close()
    
        logger.info(f"Retrieved {len(raw_papers)} arXiv papers with full metadata")
    
        return raw_papers

    # def _retrieve_raw_papers(self) -> list[ArxivResult]:
    #     client = arxiv.Client(num_retries=10, delay_seconds=10)

        
    #     # query = '+'.join(self.config.source.arxiv.category)
    #     # include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
    #     # # Get the latest paper from arxiv rss feed
    #     # feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
    #     # if 'Feed error for query' in feed.feed.title:
    #     #     raise Exception(f"Invalid ARXIV_QUERY: {query}.")
    #     # raw_papers = []
    #     # allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
    #     # all_paper_ids = [
    #     #     i.id.removeprefix("oai:arXiv.org:")
    #     #     for i in feed.entries
    #     #     if i.get("arxiv_announce_type", "new") in allowed_announce_types
    #     # ]
    #     include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
    #     allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}

    #     raw_papers = []
    #     all_paper_ids = []
    #     seen_ids = set()

    #     for category in self.config.source.arxiv.category:
    #         feed_url = f"https://rss.arxiv.org/atom/{category}"
    #         feed = feedparser.parse(feed_url)
        
    #         if getattr(feed, "bozo", False):
    #             raise Exception(
    #                 f"Failed to parse arXiv RSS feed: {feed_url}. "
    #                 f"Reason: {getattr(feed, 'bozo_exception', 'unknown')}"
    #             )
        
    #         feed_title = feed.feed.get("title", "")
    #         if "Feed error for query" in feed_title:
    #             raise Exception(f"Invalid ARXIV_QUERY: {category}.")
        
    #         for item in feed.entries:
    #             announce_type = item.get("arxiv_announce_type", "new")
    #             if announce_type not in allowed_announce_types:
    #                 continue
        
    #             paper_id = item.id.removeprefix("oai:arXiv.org:")
    #             if paper_id not in seen_ids:
    #                 seen_ids.add(paper_id)
    #                 all_paper_ids.append(paper_id)

        
    #     if self.config.executor.debug:
    #         all_paper_ids = all_paper_ids[:10]

    #     # Get full information of each paper from arxiv api
    #     bar = tqdm(total=len(all_paper_ids))
    #     max_batch_retries = 5
    #     batch_retry_delay = 30
    #     for i in range(0, len(all_paper_ids), 20):
    #         search = arxiv.Search(id_list=all_paper_ids[i:i + 20])
    #         for attempt in range(max_batch_retries):
    #             try:
    #                 batch = list(client.results(search))
    #                 bar.update(len(batch))
    #                 raw_papers.extend(batch)
    #                 break
    #             except arxiv.HTTPError as exc:
    #                 if exc.status == 429 and attempt < max_batch_retries - 1:
    #                     wait = batch_retry_delay * (attempt + 1)
    #                     logger.warning(f"arXiv API 429 on batch {i // 20}, retry {attempt + 1}/{max_batch_retries} in {wait}s")
    #                     sleep(wait)
    #                 else:
    #                     raise
    #         if i + 20 < len(all_paper_ids):
    #             sleep(3)
    #     bar.close()

    #     return raw_papers

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url
        full_text = extract_text_from_tar(raw_paper)
        if full_text is None:
            full_text = extract_text_from_html(raw_paper)
        if full_text is None:
            full_text = extract_text_from_pdf(raw_paper)
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
        )


def extract_text_from_html(paper: ArxivResult) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.entry_id, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
