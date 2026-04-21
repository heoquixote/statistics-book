import numpy as np
import pandas as pd
from scipy import stats


def build_customer_data(seed: int = 42, n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    channels = rng.choice(
        ["organic", "paid", "referral", "crm"],
        size=n,
        p=[0.35, 0.30, 0.20, 0.15],
    )
    purchase_amount = rng.lognormal(mean=3.4, sigma=0.7, size=n)
    satisfaction = rng.choice(
        [1, 2, 3, 4, 5],
        size=n,
        p=[0.06, 0.10, 0.22, 0.35, 0.27],
    )
    subscribed = rng.binomial(1, 0.42, size=n)
    response_time = rng.gamma(shape=3.0, scale=120, size=n)

    return pd.DataFrame(
        {
            "channel": channels,
            "purchase_amount": purchase_amount,
            "satisfaction": satisfaction,
            "subscribed": subscribed,
            "response_time_ms": response_time,
        }
    )


def population_vs_sample_example(df: pd.DataFrame) -> None:
    paid_only = df.loc[df["channel"] == "paid", "purchase_amount"]
    print("[1장] 모집단 vs 편의표본")
    print(f"전체 평균 구매금액: {df['purchase_amount'].mean():.2f}")
    print(f"paid 채널만 본 평균 구매금액: {paid_only.mean():.2f}")
    print()


def descriptive_stats_example(df: pd.DataFrame) -> None:
    summary = df["purchase_amount"].agg(["mean", "median", "std"])
    q1, q3 = df["purchase_amount"].quantile([0.25, 0.75])
    iqr = q3 - q1

    print("[2장] 대표값과 산포도")
    print(summary.round(2))
    print(f"IQR: {iqr:.2f}")
    print(f"왜도: {df['purchase_amount'].skew():.2f}")
    print()


def probability_example(df: pd.DataFrame) -> None:
    purchase_rate = (df["purchase_amount"] > 35).mean()
    conditional_rate = (
        df.loc[df["channel"] == "crm", "purchase_amount"] > 35
    ).mean()

    print("[3장] 기본확률과 조건부확률")
    print(f"P(구매금액 > 35): {purchase_rate:.3f}")
    print(f"P(구매금액 > 35 | channel = crm): {conditional_rate:.3f}")
    print()


def bayes_example() -> None:
    base_rate = 0.02
    recall = 0.90
    false_positive_rate = 0.08

    posterior = (recall * base_rate) / (
        (recall * base_rate) + (false_positive_rate * (1 - base_rate))
    )

    print("[3장] 베이즈 정리")
    print(f"양성 알림의 실제 정답 확률(precision 관점): {posterior:.3f}")
    print()


def clt_and_confidence_interval_example(df: pd.DataFrame, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    sample_means = []

    for _ in range(1000):
        sample = rng.choice(df["purchase_amount"], size=40, replace=True)
        sample_means.append(sample.mean())

    sample = df["purchase_amount"].sample(n=40, random_state=seed)
    mean = sample.mean()
    std = sample.std(ddof=1)
    se = std / np.sqrt(len(sample))
    ci_low, ci_high = stats.t.interval(
        confidence=0.95,
        df=len(sample) - 1,
        loc=mean,
        scale=se,
    )

    print("[4장] 중심극한정리와 신뢰구간")
    print(f"표본평균 분포 평균: {np.mean(sample_means):.2f}")
    print(f"95% 신뢰구간: ({ci_low:.2f}, {ci_high:.2f})")
    print()


def main() -> None:
    df = build_customer_data()
    population_vs_sample_example(df)
    descriptive_stats_example(df)
    probability_example(df)
    bayes_example()
    clt_and_confidence_interval_example(df)


if __name__ == "__main__":
    main()
