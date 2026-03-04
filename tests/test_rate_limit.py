"""Tests for rate-limit error classification."""

import pytest

from src.utils.rate_limit import RateLimitType, classify_error, RateLimitError


class TestClassifyError:
    """Test classify_error() against known stderr patterns."""

    def test_success_returns_none(self):
        assert classify_error("", 0) == RateLimitType.NONE
        assert classify_error("some output", 0) == RateLimitType.NONE

    # -- Weekly limit patterns --

    def test_weekly_usage_limit(self):
        assert classify_error("Error: weekly usage limit exceeded", 1) == RateLimitType.WEEKLY

    def test_weekly_cap(self):
        assert classify_error("You have reached your weekly cap", 1) == RateLimitType.WEEKLY

    def test_quota_exceeded(self):
        assert classify_error("Error: quota exceeded for this billing period", 1) == RateLimitType.WEEKLY

    def test_billing_limit(self):
        assert classify_error("billing limit reached", 1) == RateLimitType.WEEKLY

    def test_max_usage_reached(self):
        assert classify_error("max usage has been reached for your plan", 1) == RateLimitType.WEEKLY

    # -- Hourly rate limit patterns --

    def test_rate_limit(self):
        assert classify_error("Error: rate limit exceeded, please try again later", 1) == RateLimitType.HOURLY

    def test_too_many_requests(self):
        assert classify_error("too many requests", 1) == RateLimitType.HOURLY

    def test_429_error(self):
        assert classify_error("HTTP 429 Too Many Requests", 1) == RateLimitType.HOURLY

    def test_overloaded(self):
        assert classify_error("The API is currently overloaded", 1) == RateLimitType.HOURLY

    def test_throttled(self):
        assert classify_error("Request throttled", 1) == RateLimitType.HOURLY

    def test_capacity(self):
        assert classify_error("No capacity available", 1) == RateLimitType.HOURLY

    def test_try_again_later(self):
        assert classify_error("Please try again later", 1) == RateLimitType.HOURLY

    def test_resource_exhausted(self):
        assert classify_error("resource exhausted", 1) == RateLimitType.HOURLY

    def test_concurrent_limit(self):
        assert classify_error("concurrent request limit", 1) == RateLimitType.HOURLY

    # -- Transient (unknown non-zero exit) --

    def test_generic_error_is_transient(self):
        assert classify_error("some unknown error", 1) == RateLimitType.TRANSIENT

    def test_empty_stderr_nonzero(self):
        assert classify_error("", 1) == RateLimitType.TRANSIENT

    def test_none_stderr(self):
        assert classify_error(None, 1) == RateLimitType.TRANSIENT

    # -- Case insensitivity --

    def test_case_insensitive_rate_limit(self):
        assert classify_error("RATE LIMIT exceeded", 1) == RateLimitType.HOURLY

    def test_case_insensitive_weekly(self):
        assert classify_error("WEEKLY USAGE LIMIT", 1) == RateLimitType.WEEKLY


class TestRateLimitError:
    """Test the RateLimitError exception."""

    def test_is_exception(self):
        err = RateLimitError("test")
        assert isinstance(err, Exception)

    def test_default_limit_type(self):
        err = RateLimitError("test")
        assert err.limit_type == RateLimitType.WEEKLY

    def test_custom_limit_type(self):
        err = RateLimitError("test", RateLimitType.HOURLY)
        assert err.limit_type == RateLimitType.HOURLY

    def test_message(self):
        err = RateLimitError("quota exceeded")
        assert str(err) == "quota exceeded"
