import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.proportion import proportions_ztest


def build_experiment_data(seed: int = 123, n: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    group = rng.choice(["A", "B", "C"], size=n, p=[0.34, 0.33, 0.33])
    session_length = rng.normal(loc=6.5, scale=1.4, size=n)
    ad_spend_index = rng.normal(loc=50, scale=10, size=n)
    competitor_index = rng.normal(loc=40, scale=9, size=n)

    lift_map = {"A": 0.00, "B": 0.03, "C": 0.01}
    logits = -1.0 + 0.18 * session_length + np.vectorize(lift_map.get)(group)
    conversion_prob = 1 / (1 + np.exp(-logits))
    converted = rng.binomial(1, np.clip(conversion_prob, 0.01, 0.95))

    revenue = (
        120
        + 3.2 * ad_spend_index
        - 1.4 * competitor_index
        + 18 * converted
        + rng.normal(0, 20, size=n)
    )

    return pd.DataFrame(
        {
            "group": group,
            "session_length": session_length,
            "ad_spend_index": ad_spend_index,
            "competitor_index": competitor_index,
            "converted": converted,
            "revenue": revenue,
        }
    )


def hypothesis_test_example(df: pd.DataFrame) -> None:
    a = df.loc[df["group"] == "A", "revenue"]
    b = df.loc[df["group"] == "B", "revenue"]
    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)

    contingency = pd.crosstab(df["group"].isin(["A", "B"]).map({True: "AB", False: "C"}), df["converted"])

    print("[5장] Welch t-test")
    print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")
    print()
    print("[5장] 카이제곱용 교차표 예시")
    print(contingency)
    print()


def regression_example(df: pd.DataFrame) -> None:
    corr = df[["ad_spend_index", "revenue"]].corr().iloc[0, 1]

    X = sm.add_constant(df[["ad_spend_index", "competitor_index", "session_length"]])
    y = df["revenue"]
    model = sm.OLS(y, X).fit()

    vif = pd.Series(
        [
            variance_inflation_factor(X.values, i)
            for i in range(1, X.shape[1])
        ],
        index=["ad_spend_index", "competitor_index", "session_length"],
    )

    print("[6장] 상관과 회귀")
    print(f"광고지수와 매출의 Pearson 상관계수: {corr:.3f}")
    print(model.params.round(3))
    print("VIF")
    print(vif.round(3))
    print()


def ab_test_and_anova_example(df: pd.DataFrame) -> None:
    ab = df.loc[df["group"].isin(["A", "B"])].copy()
    success = ab.groupby("group")["converted"].sum().reindex(["A", "B"]).to_numpy()
    nobs = ab.groupby("group")["converted"].count().reindex(["A", "B"]).to_numpy()

    z_stat, p_value = proportions_ztest(count=success, nobs=nobs)
    f_stat, anova_p = stats.f_oneway(
        df.loc[df["group"] == "A", "revenue"],
        df.loc[df["group"] == "B", "revenue"],
        df.loc[df["group"] == "C", "revenue"],
    )

    print("[7장] A/B 테스트 z-test")
    print(f"z-statistic: {z_stat:.3f}, p-value: {p_value:.4f}")
    print()
    print("[7장] ANOVA")
    print(f"F-statistic: {f_stat:.3f}, p-value: {anova_p:.4f}")
    print()


def reporting_example(df: pd.DataFrame) -> None:
    summary = df.groupby("group")["converted"].mean().mul(100).round(2)
    lift = summary["B"] - summary["A"]

    report = (
        f"실험군 B의 전환율은 {summary['B']:.2f}%로, "
        f"대조군 A의 {summary['A']:.2f}% 대비 {lift:.2f}%p 높습니다. "
        "다만 그룹 간 유입 품질과 실험 기간 효과를 함께 검토해야 합니다."
    )

    print("[8장] 보고 문장 예시")
    print(report)
    print()


def main() -> None:
    df = build_experiment_data()
    hypothesis_test_example(df)
    regression_example(df)
    ab_test_and_anova_example(df)
    reporting_example(df)


if __name__ == "__main__":
    main()
