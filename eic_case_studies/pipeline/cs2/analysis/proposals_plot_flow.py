"""
To run the ProposalPlots flow, use the following command:

    $ python -m eic_case_studies.pipeline.cs2.analysis.proposals_plot_flow --environment pypi run

"""

# pylint: skip-file
import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table
import vl_convert as vlc
from metaflow import FlowSpec, step, Parameter, pypi_base

alt.data_transformers.disable_max_rows()


@pypi_base(
    packages={
        "pandas": "2.1.3",
        "boto3": "1.33.5",
        "pyarrow": "14.0.1",
        "altair": "4.2.2",
        "vl-convert-python": "1.3.0",
        "matplotlib": "3.8.2",
    },
    python="3.12.0",
)
class ProposalPlots(FlowSpec):

    @step
    def start(self):
        from getters.s3io import S3DataManager

        s3dm = S3DataManager()
        self.proposals_results = s3dm.load_s3_data(
            "data/05_model_output/he_2020/pathfinder/proposals/main_dbp_oa_div_components.parquet"
        )

        self.next(self.create_table)

    @step
    def create_table(self):
        """
        Create a table with the disparity components.
        """
        proposal_counts = (
            self.proposals_results.groupby(
                ["proposal_call_id", "proposal_last_evaluation_status"]
            )
            .size()
            .reset_index(name="count")
        )
        proposal_counts = (
            proposal_counts.pivot(
                index="proposal_call_id",
                columns="proposal_last_evaluation_status",
                values="count",
            )
            .fillna(0)
            .astype(int)
        )

        # Assuming proposal_counts is your DataFrame
        df = proposal_counts

        # Create a subplot with no axes
        fig, ax = plt.subplots(figsize=(15, 8))  # adjust for the size of your table
        ax.axis("off")

        # Create a table and save it as an image
        tbl = table(ax, df, loc="center", cellLoc="center")

        tbl.scale(
            1, 1.5
        )  # You can adjust the second parameter to change the row height
        tbl.auto_set_column_width(col=[20] + [6] * (len(df.columns) - 1))

        # Save the table as an image
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        plt.savefig(
            "eic_case_studies/data/06_outputs/proposals/proposal_table.png",
            dpi=300,
            bbox_inches="tight",
        )

        self.next(self.create_boxplot)

    @step
    def create_boxplot(self):

        # alt.data_transformers.enable("vegafusion")
        self.results_cp = self.proposals_results.copy()

        self.results_cp["proposal_call_deadline_date"] = pd.to_datetime(
            self.results_cp["proposal_call_deadline_date"]
        )
        order = (
            self.results_cp.groupby("proposal_call_id")["proposal_call_deadline_date"]
            .min()
            .sort_values()
            .index.tolist()
        )
        self.results_cp["contains_open"] = self.results_cp[
            "proposal_call_id"
        ].str.contains("OPEN")

        # Create a box plot for each proposal_call_id
        chart = (
            alt.Chart(self.results_cp)
            .mark_boxplot()
            .encode(
                x=alt.X("proposal_call_id:N", sort=order),
                y="div:Q",
                color=alt.Color(
                    "contains_open:N", legend=alt.Legend(title="Open Call")
                ),
            )
            .properties(width=800, height=400)
        )

        # Create a box plot for 'variety'
        chart_variety = (
            alt.Chart(self.results_cp)
            .mark_boxplot(size=6)
            .encode(
                x=alt.X("proposal_call_id:N", sort=order, axis=None),
                y="variety:Q",
                color=alt.Color(
                    "contains_open:N", legend=alt.Legend(title="Open Call")
                ),
            )
            .properties(width=270, height=170)
        )

        # Create a box plot for 'balance'
        chart_balance = (
            alt.Chart(self.results_cp)
            .mark_boxplot(size=6)
            .encode(
                x=alt.X("proposal_call_id:N", sort=order, axis=None),
                y="balance:Q",
                color=alt.Color(
                    "contains_open:N", legend=alt.Legend(title="Open Call")
                ),
            )
            .properties(width=270, height=170)
        )

        # Create a box plot for 'average_disparity'
        chart_average_disparity = (
            alt.Chart(self.results_cp)
            .mark_boxplot(size=6)
            .encode(
                x=alt.X("proposal_call_id:N", sort=order, axis=None),
                y="average_disparity:Q",
                color=alt.Color(
                    "contains_open:N", legend=alt.Legend(title="Open Call")
                ),
            )
            .properties(width=270, height=170)
        )

        # Concatenate the three charts horizontally
        combined_chart = alt.vconcat(
            chart_variety, chart_balance, chart_average_disparity
        )

        # Concatenate the combined chart with the main one vertically
        final_chart = alt.hconcat(chart, combined_chart)

        png_str = vlc.vegalite_to_png(vl_spec=final_chart.to_json(), scale=2)

        with open(
            "eic_case_studies/data/06_outputs/proposals/01_whisker_box_plot.png", "wb"
        ) as f:
            f.write(png_str)

        self.next(self.create_density_plot)

    @step
    def create_density_plot(self):
        mean_values = (
            self.results_cp.groupby("contains_open")[
                ["div", "variety", "balance", "average_disparity"]
            ]
            .mean()
            .reset_index()
        )

        # Create a density plot for 'div' with mean line
        chart = (
            alt.Chart(self.results_cp)
            .transform_density("div", as_=["div", "density"], groupby=["contains_open"])
            .mark_area(opacity=0.5)
            .encode(
                x="div:Q",
                y="density:Q",
                color=alt.Color(
                    "contains_open:N", legend=alt.Legend(title="Open Call")
                ),
            )
            .properties(width=800, height=400)
        )
        mean_line_div = (
            alt.Chart(mean_values)
            .mark_rule()
            .encode(
                x="div:Q",
                color=alt.Color(
                    "contains_open:N",
                    scale=alt.Scale(domain=[False, True], range=["#1f77b4", "#ff7f0e"]),
                ),
                size=alt.value(1),
            )
        )
        chart += mean_line_div

        # Create a density plot for 'variety' with mean line
        chart_variety = (
            alt.Chart(self.results_cp)
            .transform_density(
                "variety", as_=["variety", "density"], groupby=["contains_open"]
            )
            .mark_area(opacity=0.5)
            .encode(
                x="variety:Q",
                y="density:Q",
                color=alt.Color(
                    "contains_open:N", legend=alt.Legend(title="Open Call")
                ),
            )
            .properties(width=270, height=100)
        )
        mean_line_variety = (
            alt.Chart(mean_values)
            .mark_rule()
            .encode(
                x="variety:Q",
                color=alt.Color(
                    "contains_open:N",
                    scale=alt.Scale(domain=[False, True], range=["#1f77b4", "#ff7f0e"]),
                ),
                size=alt.value(1),
            )
        )
        chart_variety += mean_line_variety

        # Create a density plot for 'balance' with mean line
        chart_balance = (
            alt.Chart(self.results_cp)
            .transform_density(
                "balance", as_=["balance", "density"], groupby=["contains_open"]
            )
            .mark_area(opacity=0.5)
            .encode(
                x="balance:Q",
                y="density:Q",
                color=alt.Color(
                    "contains_open:N", legend=alt.Legend(title="Open Call")
                ),
            )
            .properties(width=270, height=100)
        )
        mean_line_balance = (
            alt.Chart(mean_values)
            .mark_rule()
            .encode(
                x="balance:Q",
                color=alt.Color(
                    "contains_open:N",
                    scale=alt.Scale(domain=[False, True], range=["#1f77b4", "#ff7f0e"]),
                ),
                size=alt.value(1),
            )
        )
        chart_balance += mean_line_balance

        # Create a density plot for 'average_disparity' with mean line
        chart_average_disparity = (
            alt.Chart(self.results_cp)
            .transform_density(
                "average_disparity",
                as_=["average_disparity", "density"],
                groupby=["contains_open"],
            )
            .mark_area(opacity=0.5)
            .encode(
                x="average_disparity:Q",
                y="density:Q",
                color=alt.Color(
                    "contains_open:N", legend=alt.Legend(title="Open Call")
                ),
            )
            .properties(width=270, height=100)
        )
        mean_line_average_disparity = (
            alt.Chart(mean_values)
            .mark_rule()
            .encode(
                x="average_disparity:Q",
                color=alt.Color(
                    "contains_open:N",
                    scale=alt.Scale(domain=[False, True], range=["#1f77b4", "#ff7f0e"]),
                ),
                size=alt.value(1),
            )
        )
        chart_average_disparity += mean_line_average_disparity

        # Concatenate the three charts horizontally
        combined_chart = alt.vconcat(
            chart_variety, chart_balance, chart_average_disparity
        )

        # Concatenate the combined chart with the main one vertically
        final_chart = alt.hconcat(chart, combined_chart)

        png_str = vlc.vegalite_to_png(vl_spec=final_chart.to_json(), scale=2)

        with open("eic_case_studies/data/06_outputs/proposals/02_density_plot.png", "wb") as f:
            f.write(png_str)

        self.next(self.create_wbox_status_plot)

    @step
    def create_wbox_status_plot(self):
        # Assuming 'proposals_results' is your DataFrame and it has 'proposal_last_evaluation_status', 'div', and 'proposal_call_deadline_date' columns
        self.results_cp = self.proposals_results.copy()

        # Convert 'proposal_call_deadline_date' to datetime if it's not already
        self.results_cp["proposal_call_deadline_date"] = pd.to_datetime(
            self.results_cp["proposal_call_deadline_date"]
        )

        # Create a new column 'contains_open' that indicates whether "OPEN" is included in the proposal_call_id
        self.results_cp["contains_open"] = self.results_cp[
            "proposal_call_id"
        ].str.contains("OPEN")

        # Create a box plot for each proposal_last_evaluation_status
        chart = (
            alt.Chart(self.results_cp)
            .mark_boxplot()
            .encode(
                x=alt.X("proposal_last_evaluation_status:N"),
                y="div:Q",
            )
            .properties(width=800, height=400)
        )
        # Create a box plot for 'variety'
        chart_variety = (
            alt.Chart(self.results_cp)
            .mark_boxplot(size=6)
            .encode(
                x=alt.X("proposal_last_evaluation_status:N", axis=None),
                y="variety:Q",
            )
            .properties(width=270, height=125)
        )

        # Create a box plot for 'balance'
        chart_balance = (
            alt.Chart(self.results_cp)
            .mark_boxplot(size=6)
            .encode(
                x=alt.X("proposal_last_evaluation_status:N", axis=None),
                y="balance:Q",
            )
            .properties(width=270, height=125)
        )

        # Create a box plot for 'average_disparity'
        chart_average_disparity = (
            alt.Chart(self.results_cp)
            .mark_boxplot(size=6)
            .encode(
                x=alt.X("proposal_last_evaluation_status:N", axis=None),
                y="average_disparity:Q",
            )
            .properties(width=270, height=125)
        )

        # Concatenate the three charts horizontally
        combined_chart = alt.vconcat(
            chart_variety, chart_balance, chart_average_disparity
        )

        # Concatenate the combined chart with the main one vertically
        final_chart = alt.hconcat(chart, combined_chart)

        png_str = vlc.vegalite_to_png(vl_spec=final_chart.to_json(), scale=2)

        with open(
            "eic_case_studies/data/06_outputs/proposals/03_whisker_box_status_plot.png", "wb"
        ) as f:
            f.write(png_str)

        self.next(self.create_error_bar_plots)

    @step
    def create_error_bar_plots(self):
        self.conf_intervals = self.results_cp.groupby("proposal_last_evaluation_status")[
            ["div", "variety", "balance", "average_disparity"]
        ].agg(["mean", "sem"])

        # Flatten the MultiIndex
        self.conf_intervals.columns = [
            "_".join(col).strip() for col in self.conf_intervals.columns.values
        ]

        # Reset the index
        self.conf_intervals = self.conf_intervals.reset_index()

        # Create error bars for each proposal_last_evaluation_status
        chart = alt.Chart(self.conf_intervals).mark_errorbar(extent="ci").encode(
            x=alt.X(
                "proposal_last_evaluation_status:N",
                sort=alt.EncodingSortField(
                    field="div_mean", op="mean", order="descending"
                ),
            ),
            y=alt.Y("div_mean:Q", scale=alt.Scale(zero=False)),
            yError="div_sem:Q",
        ).properties(width=800, height=400) + alt.Chart(
            self.conf_intervals
        ).mark_point().encode(
            x="proposal_last_evaluation_status:N",
            y="div_mean:Q",
            color=alt.value("black"),
        )

        # Create error bars for 'variety'
        chart_variety = alt.Chart(self.conf_intervals).mark_errorbar(extent="ci").encode(
            x=alt.X(
                "proposal_last_evaluation_status:N",
                sort=alt.EncodingSortField(
                    field="variety_mean", op="mean", order="descending"
                ),
                axis=None,
            ),
            y=alt.Y("variety_mean:Q", scale=alt.Scale(zero=False)),
            yError="variety_sem:Q",
        ).properties(width=270, height=135) + alt.Chart(
            self.conf_intervals
        ).mark_point().encode(
            x="proposal_last_evaluation_status:N",
            y="variety_mean:Q",
            color=alt.value("black"),
        )

        # Create error bars for 'balance'
        chart_balance = alt.Chart(self.conf_intervals).mark_errorbar(extent="ci").encode(
            x=alt.X(
                "proposal_last_evaluation_status:N",
                sort=alt.EncodingSortField(
                    field="balance_mean", op="mean", order="descending"
                ),
                axis=None,
            ),
            y=alt.Y("balance_mean:Q", scale=alt.Scale(zero=False)),
            yError="balance_sem:Q",
        ).properties(width=270, height=135) + alt.Chart(
            self.conf_intervals
        ).mark_point().encode(
            x="proposal_last_evaluation_status:N",
            y="balance_mean:Q",
            color=alt.value("black"),
        )

        # Create error bars for 'average_disparity'
        chart_average_disparity = alt.Chart(self.conf_intervals).mark_errorbar(
            extent="ci"
        ).encode(
            x=alt.X(
                "proposal_last_evaluation_status:N",
                sort=alt.EncodingSortField(
                    field="average_disparity_mean", op="mean", order="descending"
                ),
                axis=None,
            ),
            y=alt.Y("average_disparity_mean:Q", scale=alt.Scale(zero=False)),
            yError="average_disparity_sem:Q",
        ).properties(
            width=270, height=135
        ) + alt.Chart(
            self.conf_intervals
        ).mark_point().encode(
            x="proposal_last_evaluation_status:N",
            y="average_disparity_mean:Q",
            color=alt.value("black"),
        )

        # Concatenate the three charts horizontally
        combined_chart = alt.vconcat(
            chart_variety, chart_balance, chart_average_disparity
        )

        # Concatenate the combined chart with the main one vertically
        final_chart = alt.hconcat(chart, combined_chart)

        png_str = vlc.vegalite_to_png(vl_spec=final_chart.to_json(), scale=2)

        with open("eic_case_studies/data/06_outputs/proposals/04_error_bar_plots.png", "wb") as f:
            f.write(png_str)

        self.next(self.create_subset_error_bar_plot)

    @step
    def create_subset_error_bar_plot(self):
        self.conf_intervals = self.conf_intervals.loc[
            ~self.conf_intervals["proposal_last_evaluation_status"].isin(
                ["WITHDRAWN", "INADMISSIBLE", "INELIGIBLE"]
            )
        ]

        # Create error bars for each proposal_last_evaluation_status
        chart = alt.Chart(self.conf_intervals).mark_errorbar(extent="ci").encode(
            x=alt.X(
                "proposal_last_evaluation_status:N",
                sort=alt.EncodingSortField(
                    field="div_mean", op="mean", order="descending"
                ),
            ),
            y=alt.Y("div_mean:Q", scale=alt.Scale(zero=False)),
            yError="div_sem:Q",
        ).properties(width=800, height=400) + alt.Chart(
            self.conf_intervals
        ).mark_point().encode(
            x="proposal_last_evaluation_status:N",
            y="div_mean:Q",
            color=alt.value("black"),
        )

        # Create error bars for 'variety'
        chart_variety = alt.Chart(self.conf_intervals).mark_errorbar(extent="ci").encode(
            x=alt.X(
                "proposal_last_evaluation_status:N",
                sort=alt.EncodingSortField(
                    field="variety_mean", op="mean", order="descending"
                ),
                axis=None,
            ),
            y=alt.Y("variety_mean:Q", scale=alt.Scale(zero=False)),
            yError="variety_sem:Q",
        ).properties(width=270, height=135) + alt.Chart(
            self.conf_intervals
        ).mark_point().encode(
            x="proposal_last_evaluation_status:N",
            y="variety_mean:Q",
            color=alt.value("black"),
        )

        # Create error bars for 'balance'
        chart_balance = alt.Chart(self.conf_intervals).mark_errorbar(extent="ci").encode(
            x=alt.X(
                "proposal_last_evaluation_status:N",
                sort=alt.EncodingSortField(
                    field="balance_mean", op="mean", order="descending"
                ),
                axis=None,
            ),
            y=alt.Y("balance_mean:Q", scale=alt.Scale(zero=False)),
            yError="balance_sem:Q",
        ).properties(width=270, height=135) + alt.Chart(
            self.conf_intervals
        ).mark_point().encode(
            x="proposal_last_evaluation_status:N",
            y="balance_mean:Q",
            color=alt.value("black"),
        )

        # Create error bars for 'average_disparity'
        chart_average_disparity = alt.Chart(self.conf_intervals).mark_errorbar(
            extent="ci"
        ).encode(
            x=alt.X(
                "proposal_last_evaluation_status:N",
                sort=alt.EncodingSortField(
                    field="average_disparity_mean", op="mean", order="descending"
                ),
                axis=None,
            ),
            y=alt.Y("average_disparity_mean:Q", scale=alt.Scale(zero=False)),
            yError="average_disparity_sem:Q",
        ).properties(
            width=270, height=135
        ) + alt.Chart(
            self.conf_intervals
        ).mark_point().encode(
            x="proposal_last_evaluation_status:N",
            y="average_disparity_mean:Q",
            color=alt.value("black"),
        )

        # Concatenate the three charts horizontally
        combined_chart = alt.vconcat(
            chart_variety, chart_balance, chart_average_disparity
        )

        # Concatenate the combined chart with the main one vertically
        final_chart = alt.hconcat(chart, combined_chart)

        png_str = vlc.vegalite_to_png(vl_spec=final_chart.to_json(), scale=2)

        with open(
            "eic_case_studies/data/06_outputs/proposals/05_subset_error_bar_plot.png", "wb"
        ) as f:
            f.write(png_str)

        self.next(self.end)

    @step
    def end(self):
        """
        End the flow.
        """
        pass


if __name__ == "__main__":
    ProposalPlots()
