"""End-to-end tests for the observability of redirection handling.

A redirection chain that loops back to a URL it already visited must not be
silently swallowed by the duplicate request filter: it raises
RedirectLoopError, which reaches the request errback and increases the
redirect/loop_count stat. A chain whose URLs keep changing still fails with
IgnoreRequest once REDIRECT_MAX_TIMES is reached, and ordinary duplicate
requests are still filtered by the dupefilter without inflating any
redirect-specific stat.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from scrapy.exceptions import IgnoreRequest, RedirectLoopError
from scrapy.http import Request
from scrapy.spiders import Spider
from scrapy.utils.test import get_crawler
from tests.utils.decorators import coroutine_test

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from twisted.python.failure import Failure

    from tests.mockserver.http import MockServer


class _RedirectObservabilitySpider(Spider):
    name = "redirect_observability"

    def __init__(self, *args: Any, start_urls: list[str], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._start_urls = start_urls
        self.loop_failures: list[Failure] = []
        self.max_redirect_failures: list[Failure] = []

    async def start(self) -> AsyncIterator[Request]:
        for url in self._start_urls:
            yield Request(url, errback=self.errback, dont_filter=True)

    def errback(self, failure: Failure) -> None:
        if failure.check(RedirectLoopError):
            self.loop_failures.append(failure)
        elif failure.check(IgnoreRequest):
            self.max_redirect_failures.append(failure)
        else:
            assert failure.value is not None
            raise failure.value


@coroutine_test
async def test_redirect_loop_reaches_errback_and_stat(
    mockserver: MockServer,
) -> None:
    crawler = get_crawler(_RedirectObservabilitySpider)
    await crawler.crawl_async(
        mockserver=mockserver,
        start_urls=[mockserver.url("/redirect-loop/a")],
    )
    spider = cast("_RedirectObservabilitySpider", crawler.spider)
    assert spider is not None

    # The loop is surfaced through the errback, not swallowed.
    assert len(spider.loop_failures) == 1
    failure = spider.loop_failures[0]
    assert isinstance(failure.value, RedirectLoopError)
    # The looped-back URL is exposed on the exception.
    assert failure.value.url.endswith("/redirect-loop/a")

    stats = crawler.stats
    assert stats.get_value("redirect/loop_count") == 1
    # The loop never reaches the dupefilter, so its stat is not inflated.
    assert stats.get_value("dupefilter/filtered") is None
    # The max-redirections path is not involved.
    assert spider.max_redirect_failures == []


@coroutine_test
async def test_max_redirect_times_still_raises(mockserver: MockServer) -> None:
    crawler = get_crawler(_RedirectObservabilitySpider, {"REDIRECT_MAX_TIMES": 3})
    await crawler.crawl_async(
        mockserver=mockserver,
        start_urls=[mockserver.url("/infinite-redirect/0")],
    )
    spider = cast("_RedirectObservabilitySpider", crawler.spider)
    assert spider is not None

    # The ever-changing chain still hits the max-redirections error.
    assert len(spider.max_redirect_failures) == 1
    assert spider.max_redirect_failures[0].check(IgnoreRequest)
    assert not spider.max_redirect_failures[0].check(RedirectLoopError)

    stats = crawler.stats
    assert stats.get_value("redirect/loop_count") is None
    assert stats.get_value("dupefilter/filtered") is None


class _DuplicateRequestSpider(Spider):
    """Yields the same ordinary URL twice, which the dupefilter must drop."""

    name = "duplicate_request"

    def __init__(self, *args: Any, duplicated_url: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._duplicated_url = duplicated_url
        self.seen: set[str] = set()

    async def start(self) -> AsyncIterator[Request]:
        yield Request(self._duplicated_url)
        yield Request(self._duplicated_url)

    def parse(self, response: Any) -> None:
        self.seen.add(response.url)


@coroutine_test
async def test_ordinary_duplicates_still_filtered(mockserver: MockServer) -> None:
    crawler = get_crawler(_DuplicateRequestSpider)
    await crawler.crawl_async(
        mockserver=mockserver,
        duplicated_url=mockserver.url("/status?n=200"),
    )
    stats = crawler.stats
    # The ordinary duplicate is counted by the dupefilter as before.
    assert stats.get_value("dupefilter/filtered") == 1
    # And no redirect-loop stat is created for it.
    assert stats.get_value("redirect/loop_count") is None
