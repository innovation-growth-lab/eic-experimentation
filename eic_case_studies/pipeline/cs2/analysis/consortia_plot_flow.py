"""
To run the ResearcherPlots flow, use the following command:

    $ python -m eic_case_studies.pipeline.cs2.analysis.consortia_plot_flow --environment pypi run

"""

# pylint: skip-file
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
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
class ResearcherConsortiaPlots(FlowSpec):

    @step
    def start(self):
        from getters.s3io import S3DataManager

        s3dm = S3DataManager()
        self.researchers_results = s3dm.load_s3_data(
            "data/05_model_output/he_2020/pathfinder/roles/consortia_outputs_agg.parquet"
        )

        self.next(self.create_table)

    @step
    def create_table(self):
        """
        Create a table with the disparity components.
        """

        print("ha")
        researcher_counts = (
            self.researchers_results.loc[self.researchers_results["publication_year"]=="all"].groupby(["proposal_call_id", "status"])
            .size()
            .reset_index(name="count")
        )
        researcher_counts = (
            researcher_counts.pivot(
                index="proposal_call_id",
                columns="status",
                values="count",
            )
            .fillna(0)
            .astype(int)
        )

        # Assuming researcher_counts is your DataFrame
        df = researcher_counts

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
            "eic_case_studies/data/06_outputs/consortia/00_consortia_table.png",
            dpi=400,
            bbox_inches="tight",
        )

        self.next(self.create_boxplot)

    @step
    def create_boxplot(self):

        self.results_cp = self.researchers_results.copy()

        # Extract the year from proposal_call_id
        self.results_cp["proposal_call_year"] = self.results_cp[
            "proposal_call_id"
        ].str.extract(r"(\d{4})")

        # Create an ordered list of proposal_call_id values, sorted by year
        order = (
            self.results_cp.sort_values("proposal_call_year")["proposal_call_id"]
            .unique()
            .tolist()
        )

        self.results_cp["contains_open"] = self.results_cp[
            "proposal_call_id"
        ].str.contains("OPEN")

        # Create a box plot for each proposal_call_id
        create_whisker_boxplot(order, "all", self.results_cp)
        create_whisker_boxplot(order, "before", self.results_cp)
        create_whisker_boxplot(order, "after", self.results_cp)

        self.next(self.create_density_plot)

    @step
    def create_density_plot(self):
        mean_values = (
            self.results_cp.groupby(["status", "publication_year"])[
                ["div", "variety", "balance", "average_disparity"]
            ]
            .mean()
            .reset_index()
        )

        # Create a density plot for all publication_years
        density_plot(self.results_cp, mean_values, "all")
        density_plot(self.results_cp, mean_values, "before")

        self.next(self.create_wbox_status_plot)

    @step
    def create_wbox_status_plot(self):
        # Assuming 'researchers_results' is your DataFrame and it has 'researcher_last_evaluation_status', 'div', and 'researcher_call_deadline_date' columns
        self.results_cp = self.researchers_results.copy()

        create_wbox_status(self.results_cp, "all")

        create_wbox_status(self.results_cp, "All - EIC")

        self.next(self.create_error_bar_plots)

    @step
    def create_error_bar_plots(self):

        # Create separate dataframes for 'before' and 'after'
        self.results_cp_before = self.results_cp[
            self.results_cp["publication_year"] == "before"
        ]
        self.results_cp_after = self.results_cp[
            self.results_cp["publication_year"] == "after"
        ]

        # Calculate confidence intervals for 'before'
        self.conf_intervals_before = self.results_cp_before.groupby("status")[
            ["div", "variety", "balance", "average_disparity"]
        ].agg(["mean", "sem"])

        # Flatten the MultiIndex
        self.conf_intervals_before.columns = [
            "_".join(col).strip() for col in self.conf_intervals_before.columns.values
        ]
        # Reset the index
        self.conf_intervals_before = self.conf_intervals_before.reset_index()

        # Calculate confidence intervals for 'after'
        self.conf_intervals_after = self.results_cp_after.groupby("status")[
            ["div", "variety", "balance", "average_disparity"]
        ].agg(["mean", "sem"])

        # Flatten the MultiIndex
        self.conf_intervals_after.columns = [
            "_".join(col).strip() for col in self.conf_intervals_after.columns.values
        ]

        # Reset the index
        self.conf_intervals_after = self.conf_intervals_after.reset_index()

        # Add a 'time_period' column to each dataframe
        self.conf_intervals_before["time_period"] = "before"
        self.conf_intervals_after["time_period"] = "after"

        # Concatenate the dataframes
        self.conf_intervals = pd.concat(
            [self.conf_intervals_before, self.conf_intervals_after]
        )

        # Create the error bars
        error_bars = (
            alt.Chart(self.conf_intervals)
            .mark_errorbar(extent="ci")
            .encode(
                x=alt.X("time_period:N", sort=["before", "after"], title=None),
                y=alt.Y("div_mean:Q", title="Diversity", scale=alt.Scale(zero=False)),
                yError="div_sem:Q",
                color="time_period:N",
            )
        )

        # Create the points for the mean
        points = (
            alt.Chart(self.conf_intervals)
            .mark_point(filled=True)
            .encode(
                x=alt.X("time_period:N", sort=["before", "after"], title=None),
                y=alt.Y("div_mean:Q", title="Diversity", scale=alt.Scale(zero=False)),
                color="time_period:N",
            )
        )

        # Combine the error bars and points
        final_chart = (error_bars + points).facet(column="status:N")

        png_str = vlc.vegalite_to_png(vl_spec=final_chart.to_json(), scale=2)

        with open(
            "eic_case_studies/data/06_outputs/consortia/04_error_bar_plots.png", "wb"
        ) as f:
            f.write(png_str)

        self.next(self.create_error_bar_plots_citations)

    @step
    def create_error_bar_plots_citations(self):

        self.results_cp = self.researchers_results.copy()
        # Create separate dataframes for 'before' and 'after'
        self.results_cp_before = self.results_cp[
            self.results_cp["publication_year"] == "before"
        ]
        self.results_cp_after = self.results_cp[
            self.results_cp["publication_year"] == "after"
        ]

        # Calculate mean of 'cited_by_count_std' for 'before' and 'after'
        self.results_cp_before["cited_by_count_std_mean"] = self.results_cp_before[
            "cited_by_count_std"
        ].apply(lambda x: np.mean(x))
        self.results_cp_after["cited_by_count_std_mean"] = self.results_cp_after[
            "cited_by_count_std"
        ].apply(lambda x: np.mean(x))

        # Calculate confidence intervals for 'before'
        self.conf_intervals_before = self.results_cp_before.groupby("status")[
            ["cited_by_count_std_mean"]
        ].agg(["mean", "sem"])

        # Flatten the MultiIndex
        self.conf_intervals_before.columns = [
            "_".join(col).strip() for col in self.conf_intervals_before.columns.values
        ]
        # Reset the index
        self.conf_intervals_before = self.conf_intervals_before.reset_index()

        # Calculate confidence intervals for 'after'
        self.conf_intervals_after = self.results_cp_after.groupby("status")[
            ["cited_by_count_std_mean"]
        ].agg(["mean", "sem"])

        # Flatten the MultiIndex
        self.conf_intervals_after.columns = [
            "_".join(col).strip() for col in self.conf_intervals_after.columns.values
        ]

        # Reset the index
        self.conf_intervals_after = self.conf_intervals_after.reset_index()

        # Add a 'time_period' column to each dataframe
        self.conf_intervals_before["time_period"] = "before"
        self.conf_intervals_after["time_period"] = "after"

        # Concatenate the dataframes
        self.conf_intervals = pd.concat(
            [self.conf_intervals_before, self.conf_intervals_after]
        )

        # Create the error bars
        error_bars = (
            alt.Chart(self.conf_intervals)
            .mark_errorbar(extent="ci")
            .encode(
                x=alt.X("time_period:N", sort=["before", "after"], title=None),
                y=alt.Y(
                    "cited_by_count_std_mean_mean:Q",
                    title="Average Standardized Citation Count",
                    scale=alt.Scale(zero=False),
                ),
                yError="cited_by_count_std_mean_sem:Q",
                color="time_period:N",
            )
        )

        # Create the points for the mean
        points = (
            alt.Chart(self.conf_intervals)
            .mark_point(filled=True)
            .encode(
                x=alt.X("time_period:N", sort=["before", "after"], title=None),
                y=alt.Y(
                    "cited_by_count_std_mean_mean:Q",
                    title="Average Standardized Citation Count",
                    scale=alt.Scale(zero=False),
                ),
                color="time_period:N",
            )
        )

        # Combine the error bars and points
        final_chart = (error_bars + points).facet(column="status:N")

        png_str = vlc.vegalite_to_png(vl_spec=final_chart.to_json(), scale=2)

        with open(
            "eic_case_studies/data/06_outputs/consortia/04_error_bar_plots_citations.png",
            "wb",
        ) as f:
            f.write(png_str)

        self.next(self.create_error_bar_plots_publications)

    @step
    def create_error_bar_plots_publications(self):

        self.results_cp = self.researchers_results.copy()
        # Create separate dataframes for 'before' and 'after'
        self.results_cp_before = self.results_cp[
            self.results_cp["publication_year"] == "before"
        ]
        self.results_cp_after = self.results_cp[
            self.results_cp["publication_year"] == "after"
        ]

        # Calculate mean of 'publications_count_std' for 'before' and 'after'
        self.results_cp_before["publications_count_std_mean"] = self.results_cp_before[
            "publications_count_std"
        ].apply(lambda x: np.mean(x))
        self.results_cp_after["publications_count_std_mean"] = self.results_cp_after[
            "publications_count_std"
        ].apply(lambda x: np.mean(x))

        # Calculate confidence intervals for 'before'
        self.conf_intervals_before = self.results_cp_before.groupby("status")[
            ["publications_count_std_mean"]
        ].agg(["mean", "sem"])

        # Flatten the MultiIndex
        self.conf_intervals_before.columns = [
            "_".join(col).strip() for col in self.conf_intervals_before.columns.values
        ]
        # Reset the index
        self.conf_intervals_before = self.conf_intervals_before.reset_index()

        # Calculate confidence intervals for 'after'
        self.conf_intervals_after = self.results_cp_after.groupby("status")[
            ["publications_count_std_mean"]
        ].agg(["mean", "sem"])

        # Flatten the MultiIndex
        self.conf_intervals_after.columns = [
            "_".join(col).strip() for col in self.conf_intervals_after.columns.values
        ]

        # Reset the index
        self.conf_intervals_after = self.conf_intervals_after.reset_index()

        # Add a 'time_period' column to each dataframe
        self.conf_intervals_before["time_period"] = "before"
        self.conf_intervals_after["time_period"] = "after"

        # Concatenate the dataframes
        self.conf_intervals = pd.concat(
            [self.conf_intervals_before, self.conf_intervals_after]
        )

        # Create the error bars
        error_bars = (
            alt.Chart(self.conf_intervals)
            .mark_errorbar(extent="ci")
            .encode(
                x=alt.X("time_period:N", sort=["before", "after"], title=None),
                y=alt.Y(
                    "publications_count_std_mean_mean:Q",
                    title="Average Standardized Publication Count",
                    scale=alt.Scale(zero=False),
                ),
                yError="publications_count_std_mean_sem:Q",
                color="time_period:N",
            )
        )

        # Create the points for the mean
        points = (
            alt.Chart(self.conf_intervals)
            .mark_point(filled=True)
            .encode(
                x=alt.X("time_period:N", sort=["before", "after"], title=None),
                y=alt.Y(
                    "publications_count_std_mean_mean:Q",
                    title="Average Standardized Publication Count",
                    scale=alt.Scale(zero=False),
                ),
                color="time_period:N",
            )
        )

        # Combine the error bars and points
        final_chart = (error_bars + points).facet(column="status:N")

        png_str = vlc.vegalite_to_png(vl_spec=final_chart.to_json(), scale=2)

        with open(
            "eic_case_studies/data/06_outputs/consortia/04_error_bar_plots_publications.png",
            "wb",
        ) as f:
            f.write(png_str)

        self.next(self.create_density_subset)

    @step
    def create_density_subset(self):

        subsets = ["MAIN", "NO_MONEY", "REJECTED", "NOT_EIC"]
        self.results_cp = self.researchers_results.copy().dropna(subset=["div"])
        self.results_cp = self.results_cp[self.results_cp["status"].isin(subsets)]
        self.results_cp = self.results_cp.loc[
            self.results_cp["publication_year"] == "all"
        ]
        self.results_cp = self.results_cp[self.results_cp["div"] <= 0.005]

        # Initialize an empty list to hold the plots
        plots = []

        # Create a density plot for each status class
        for status in subsets:
            # Create the area plot
            area_plot = (
                alt.Chart(self.results_cp[self.results_cp["status"] == status])
                .transform_density(
                    "div", as_=["div", "density"], extent=[0, 0.005], groupby=["status"]
                )
                .mark_area(opacity=0.2)
                .encode(x="div:Q", y="density:Q", color="status:N")
            )

            # Create the line plot
            line_plot = (
                alt.Chart(self.results_cp[self.results_cp["status"] == status])
                .transform_density(
                    "div", as_=["div", "density"], extent=[0, 0.005], groupby=["status"]
                )
                .mark_line(opacity=0.5, size=3)
                .encode(alt.X("div:Q"), alt.Y("density:Q"), alt.Color("status:N"))
            )

            # Layer the area and line plots
            plot = alt.layer(area_plot, line_plot)

            plots.append(plot)

        # Overlay the plots
        overlay_plot = alt.layer(*plots)

        png_str = vlc.vegalite_to_png(vl_spec=overlay_plot.to_json(), scale=2)

        with open(
            "eic_case_studies/data/06_outputs/consortia/05_density_subplot.png", "wb"
        ) as f:
            f.write(png_str)

        # do the same now with cited_by_count_std
        self.results_cp = self.researchers_results.copy()
        self.results_cp = self.results_cp[
            self.results_cp["status"].isin(["MAIN", "NO_MONEY", "REJECTED", "NOT_EIC"])
        ]
        self.results_cp = self.results_cp.loc[
            self.results_cp["publication_year"] == "all"
        ]
        self.results_cp["cited_by_count_std_mean"] = self.results_cp[
            "cited_by_count_std"
        ].apply(lambda x: np.mean(x))
        self.results_cp = self.results_cp.copy().dropna(
            subset=["cited_by_count_std_mean"]
        )

        # Initialize an empty list to hold the plots
        plots = []

        for status in subsets:
            # Create the area plot
            area_plot = (
                alt.Chart(self.results_cp[self.results_cp["status"] == status])
                .transform_density(
                    "cited_by_count_std_mean",
                    as_=["cited_by_count_std_mean", "density"],
                    extent=[-2, 6],
                    groupby=["status"],
                )
                .mark_area(opacity=0.2)
                .encode(x="cited_by_count_std_mean:Q", y="density:Q", color="status:N")
            )

            # Create the line plot
            line_plot = (
                alt.Chart(self.results_cp[self.results_cp["status"] == status])
                .transform_density(
                    "cited_by_count_std_mean",
                    as_=["cited_by_count_std_mean", "density"],
                    extent=[-2, 6],
                    groupby=["status"],
                )
                .mark_line(opacity=0.5, size=3)
                .encode(
                    alt.X("cited_by_count_std_mean:Q"),
                    alt.Y("density:Q"),
                    alt.Color("status:N"),
                )
            )

            # Layer the area and line plots
            plot = alt.layer(area_plot, line_plot)

            plots.append(plot)

        # Overlay the plots
        overlay_plot = alt.layer(*plots)

        png_str = vlc.vegalite_to_png(vl_spec=overlay_plot.to_json(), scale=2)

        with open(
            "eic_case_studies/data/06_outputs/consortia/05_density_subplot_cited_by_count_std.png",
            "wb",
        ) as f:
            f.write(png_str)

        # do the same now with publication_count_std
        self.results_cp = self.researchers_results.copy()
        self.results_cp = self.results_cp[
            self.results_cp["status"].isin(["MAIN", "NO_MONEY", "REJECTED", "NOT_EIC"])
        ]
        self.results_cp = self.results_cp.loc[
            self.results_cp["publication_year"] == "all"
        ]
        self.results_cp["publications_count_std_mean"] = self.results_cp[
            "publications_count_std"
        ].apply(lambda x: np.mean(x))
        self.results_cp = self.results_cp.copy().dropna(
            subset=["publications_count_std_mean"]
        )

        # Initialize an empty list to hold the plots
        plots = []

        for status in subsets:
            # Create the area plot
            area_plot = (
                alt.Chart(self.results_cp[self.results_cp["status"] == status])
                .transform_density(
                    "publications_count_std_mean",
                    as_=["publications_count_std_mean", "density"],
                    extent=[-2, 6],
                    groupby=["status"],
                )
                .mark_area(opacity=0.2)
                .encode(
                    x="publications_count_std_mean:Q", y="density:Q", color="status:N"
                )
            )

            # Create the line plot
            line_plot = (
                alt.Chart(self.results_cp[self.results_cp["status"] == status])
                .transform_density(
                    "publications_count_std_mean",
                    as_=["publications_count_std_mean", "density"],
                    extent=[-2, 2],
                    groupby=["status"],
                )
                .mark_line(opacity=0.5, size=3)
                .encode(
                    alt.X("publications_count_std_mean:Q"),
                    alt.Y("density:Q"),
                    alt.Color("status:N"),
                )
            )

            # Layer the area and line plots
            plot = alt.layer(area_plot, line_plot)

            plots.append(plot)

        # Overlay the plots
        overlay_plot = alt.layer(*plots)

        png_str = vlc.vegalite_to_png(vl_spec=overlay_plot.to_json(), scale=2)

        with open(
            "eic_case_studies/data/06_outputs/consortia/05_density_subplot_publication_count_std.png",
            "wb",
        ) as f:
            f.write(png_str)

        self.next(self.create_publications_scatter)

    @step
    def create_publications_scatter(self):
        # Calculate 'publication_count_std_mean' for 'before' and 'after'
        self.results_cp_before = self.researchers_results.copy()
        self.results_cp_before = self.results_cp_before[
            self.results_cp_before["status"].isin(
                ["MAIN", "NO_MONEY", "REJECTED", "NOT_EIC"]
            )
        ]
        self.results_cp_before = self.results_cp_before.loc[
            self.results_cp_before["publication_year"] == "before"
        ]
        self.results_cp_before["publications_count_std_mean"] = self.results_cp_before[
            "publications_count_std"
        ].apply(lambda x: np.mean(x))

        self.results_cp_after = self.researchers_results.copy()
        self.results_cp_after = self.results_cp_after[
            self.results_cp_after["status"].isin(
                ["MAIN", "NO_MONEY", "REJECTED", "NOT_EIC"]
            )
        ]
        self.results_cp_after = self.results_cp_after.loc[
            self.results_cp_after["publication_year"] == "after"
        ]
        self.results_cp_after["publications_count_std_mean"] = self.results_cp_after[
            "publications_count_std"
        ].apply(lambda x: np.mean(x))

        # Merge the two dataframes on researcher id
        self.results_cp = pd.merge(
            self.results_cp_before,
            self.results_cp_after[["researcher", "publications_count_std_mean"]],
            on="researcher",
            suffixes=("_before", "_after"),
        )

        # Create the scatter plot
        scatter_plot = (
            alt.Chart(self.results_cp)
            .mark_circle(size=60)
            .encode(
                x=alt.X(
                    "publications_count_std_mean_before:Q",
                    title="Publications Count Std Mean Before",
                ),
                y=alt.Y(
                    "publications_count_std_mean_after:Q",
                    title="Publications Count Std Mean After",
                ),
                color=alt.Color("status:N", legend=alt.Legend(title="Status")),
                tooltip=[
                    "status:N",
                    "publications_count_std_mean_before:Q",
                    "publications_count_std_mean_after:Q",
                ],
            )
            .properties(
                title="Comparison of Publications Count Std Mean Before and After",
                width=600,
                height=400,
            )
        )

        # Create the fitted lines
        fitted_lines = scatter_plot.transform_regression(
            "publications_count_std_mean_before",
            "publications_count_std_mean_after",
            groupby=["status"],
        ).mark_line()

        # Create a 45-degree line
        degree_line = (
            alt.Chart(
                pd.DataFrame(
                    {
                        "x": [
                            self.results_cp["publications_count_std_mean_before"].min(),
                            self.results_cp["publications_count_std_mean_before"].max(),
                        ]
                    }
                )
            )
            .mark_line(color="black", strokeDash=[3, 3])
            .encode(x="x", y="x")
        )

        # Combine the scatter plot, the fitted lines, and the 45-degree line
        final_plot = scatter_plot + fitted_lines + degree_line

        # Save the plot as a PNG file
        png_str = vlc.vegalite_to_png(vl_spec=final_plot.to_json(), scale=2)

        with open(
            "eic_case_studies/data/06_outputs/consortia/06_scatterplot_publications_count_std.png",
            "wb",
        ) as f:
            f.write(png_str)

        self.next(self.create_citations_scatter)

    @step
    def create_citations_scatter(self):
        # Calculate 'cited_by_count_mean' for 'before' and 'after'
        self.results_cp_before = self.researchers_results.copy()
        self.results_cp_before = self.results_cp_before[
            self.results_cp_before["status"].isin(
                ["MAIN", "NO_MONEY", "REJECTED", "NOT_EIC"]
            )
        ]
        self.results_cp_before = self.results_cp_before.loc[
            self.results_cp_before["publication_year"] == "before"
        ]
        self.results_cp_before["cited_by_count_mean"] = self.results_cp_before[
            "cited_by_count_std"
        ].apply(lambda x: np.mean(x))

        self.results_cp_after = self.researchers_results.copy()
        self.results_cp_after = self.results_cp_after[
            self.results_cp_after["status"].isin(
                ["MAIN", "NO_MONEY", "REJECTED", "NOT_EIC"]
            )
        ]
        self.results_cp_after = self.results_cp_after.loc[
            self.results_cp_after["publication_year"] == "after"
        ]
        self.results_cp_after["cited_by_count_mean"] = self.results_cp_after[
            "cited_by_count_std"
        ].apply(lambda x: np.mean(x))

        # Merge the two dataframes on researcher id
        self.results_cp = pd.merge(
            self.results_cp_before,
            self.results_cp_after[["researcher", "cited_by_count_mean"]],
            on="researcher",
            suffixes=("_before", "_after"),
        )

        # Create the scatter plot
        scatter_plot = (
            alt.Chart(self.results_cp)
            .mark_circle(size=60)
            .encode(
                x=alt.X(
                    "cited_by_count_mean_before:Q", title="Cited By Count Mean Before"
                ),
                y=alt.Y(
                    "cited_by_count_mean_after:Q", title="Cited By Count Mean After"
                ),
                color=alt.Color("status:N", legend=alt.Legend(title="Status")),
                tooltip=[
                    "status:N",
                    "cited_by_count_mean_before:Q",
                    "cited_by_count_mean_after:Q",
                ],
            )
            .properties(
                title="Comparison of Cited By Count Mean Before and After",
                width=600,
                height=400,
            )
        )

        # Create the fitted lines
        fitted_lines = scatter_plot.transform_regression(
            "cited_by_count_mean_before",
            "cited_by_count_mean_after",
            groupby=["status"],
        ).mark_line()

        # Create a 45-degree line
        degree_line = (
            alt.Chart(
                pd.DataFrame(
                    {
                        "x": [
                            self.results_cp["cited_by_count_mean_before"].min(),
                            self.results_cp["cited_by_count_mean_before"].max(),
                        ]
                    }
                )
            )
            .mark_line(color="black", strokeDash=[3, 3])
            .encode(x="x", y="x")
        )

        # Combine the scatter plot, the fitted lines, and the 45-degree line
        final_plot = scatter_plot + fitted_lines + degree_line

        # Save the plot as a PNG file
        png_str = vlc.vegalite_to_png(vl_spec=final_plot.to_json(), scale=2)

        with open(
            "eic_case_studies/data/06_outputs/consortia/06_scatterplot_cited_by_count_mean.png",
            "wb",
        ) as f:
            f.write(png_str)

        self.next(self.create_scatter_citations_div)

    @step
    def create_scatter_citations_div(self):
        # Calculate 'div_mean' for 'before' and 'cited_by_count_mean' for 'after'
        self.results_cp_before = self.researchers_results.copy()
        self.results_cp_before = self.results_cp_before[
            self.results_cp_before["status"].isin(
                ["MAIN", "NO_MONEY", "REJECTED", "NOT_EIC"]
            )
        ]
        self.results_cp_before = self.results_cp_before.loc[
            self.results_cp_before["publication_year"] == "before"
        ]
        self.results_cp_before["div_mean"] = self.results_cp_before["div"].apply(
            lambda x: np.mean(x)
        )

        self.results_cp_after = self.researchers_results.copy()
        self.results_cp_after = self.results_cp_after[
            self.results_cp_after["status"].isin(
                ["MAIN", "NO_MONEY", "REJECTED", "NOT_EIC"]
            )
        ]
        self.results_cp_after = self.results_cp_after.loc[
            self.results_cp_after["publication_year"] == "after"
        ]
        self.results_cp_after["cited_by_count_mean"] = self.results_cp_after[
            "cited_by_count_std"
        ].apply(lambda x: np.mean(x))

        # Merge the two dataframes on researcher id
        self.results_cp = pd.merge(
            self.results_cp_before,
            self.results_cp_after[["researcher", "cited_by_count_mean"]],
            on="researcher",
            suffixes=("_before", "_after"),
        )

        # Create the scatter plot
        scatter_plot = (
            alt.Chart(self.results_cp)
            .mark_circle(size=60)
            .encode(
                x=alt.X("div_mean:Q", title="Div Mean Before"),
                y=alt.Y("cited_by_count_mean:Q", title="Cited By Count Mean After"),
                color=alt.Color("status:N", legend=alt.Legend(title="Status")),
                tooltip=["status:N", "div_mean:Q", "cited_by_count_mean:Q"],
            )
            .properties(
                title="Comparison of Div Mean Before and Cited By Count Mean After",
                width=600,
                height=400,
            )
        )

        # Create the fitted lines
        fitted_lines = scatter_plot.transform_regression(
            "div_mean", "cited_by_count_mean", groupby=["status"]
        ).mark_line()

        # Create a 45-degree line
        degree_line = (
            alt.Chart(
                pd.DataFrame(
                    {
                        "x": [
                            self.results_cp["div_mean"].min(),
                            self.results_cp["div_mean"].max(),
                        ]
                    }
                )
            )
            .mark_line(color="black", strokeDash=[3, 3])
            .encode(x="x", y="x")
        )

        # Combine the scatter plot, the fitted lines, and the 45-degree line
        final_plot = scatter_plot + fitted_lines + degree_line

        # Save the plot as a PNG file
        png_str = vlc.vegalite_to_png(vl_spec=final_plot.to_json(), scale=2)

        with open(
            "eic_case_studies/data/06_outputs/consortia/06_scatterplot_div_mean_cited_by_count_mean.png",
            "wb",
        ) as f:
            f.write(png_str)

        self.next(self.create_scatter_publications_div)

    @step
    def create_scatter_publications_div(self):
        # Calculate 'div_mean' for 'before' and 'publications_count_std' for 'after'
        self.results_cp_before = self.researchers_results.copy()
        self.results_cp_before = self.results_cp_before[
            self.results_cp_before["status"].isin(
                ["MAIN", "NO_MONEY", "REJECTED", "NOT_EIC"]
            )
        ]
        self.results_cp_before = self.results_cp_before.loc[
            self.results_cp_before["publication_year"] == "before"
        ]
        self.results_cp_before["div_mean"] = self.results_cp_before["div"].apply(
            lambda x: np.mean(x)
        )

        self.results_cp_after = self.researchers_results.copy()
        self.results_cp_after = self.results_cp_after[
            self.results_cp_after["status"].isin(
                ["MAIN", "NO_MONEY", "REJECTED", "NOT_EIC"]
            )
        ]
        self.results_cp_after = self.results_cp_after.loc[
            self.results_cp_after["publication_year"] == "after"
        ]

        # Merge the two dataframes on researcher id
        self.results_cp = pd.merge(
            self.results_cp_before,
            self.results_cp_after[["researcher", "publications_count_std"]],
            on="researcher",
            suffixes=("_before", "_after"),
        )

        # Create the scatter plot
        scatter_plot = (
            alt.Chart(self.results_cp)
            .mark_circle(size=60)
            .encode(
                x=alt.X("div_mean:Q", title="Div Mean Before"),
                y=alt.Y(
                    "publications_count_std_after:Q",
                    title="Publications Count Std After",
                ),
                color=alt.Color("status:N", legend=alt.Legend(title="Status")),
                tooltip=["status:N", "div_mean:Q", "publications_count_std_after:Q"],
            )
            .properties(
                title="Comparison of Div Mean Before and Publications Count Std After",
                width=600,
                height=400,
            )
        )

        # Create the fitted lines
        fitted_lines = scatter_plot.transform_regression(
            "div_mean", "publications_count_std_after", groupby=["status"]
        ).mark_line()

        # Create a 45-degree line
        degree_line = (
            alt.Chart(
                pd.DataFrame(
                    {
                        "x": [
                            self.results_cp["div_mean"].min(),
                            self.results_cp["div_mean"].max(),
                        ]
                    }
                )
            )
            .mark_line(color="black", strokeDash=[3, 3])
            .encode(x="x", y="x")
        )

        # Combine the scatter plot, the fitted lines, and the 45-degree line
        final_plot = scatter_plot + fitted_lines + degree_line

        # Save the plot as a PNG file
        png_str = vlc.vegalite_to_png(vl_spec=final_plot.to_json(), scale=2)

        with open(
            "eic_case_studies/data/06_outputs/consortia/06_scatterplot_div_mean_publications_count_std.png",
            "wb",
        ) as f:
            f.write(png_str)

        self.next(self.create_scatter_div_before_after)

    @step
    def create_scatter_div_before_after(self):
        # Calculate 'div_mean' for 'before' and 'after'
        self.results_cp_before = self.researchers_results.copy()
        self.results_cp_before = self.results_cp_before[
            self.results_cp_before["status"].isin(
                ["MAIN", "NO_MONEY", "REJECTED", "NOT_EIC"]
            )
        ]
        self.results_cp_before = self.results_cp_before.loc[
            self.results_cp_before["publication_year"] == "before"
        ]
        self.results_cp_before["div_mean"] = self.results_cp_before["div"].apply(
            lambda x: np.mean(x)
        )

        self.results_cp_after = self.researchers_results.copy()
        self.results_cp_after = self.results_cp_after[
            self.results_cp_after["status"].isin(
                ["MAIN", "NO_MONEY", "REJECTED", "NOT_EIC"]
            )
        ]
        self.results_cp_after = self.results_cp_after.loc[
            self.results_cp_after["publication_year"] == "after"
        ]
        self.results_cp_after["div_mean"] = self.results_cp_after["div"].apply(
            lambda x: np.mean(x)
        )

        # Merge the two dataframes on researcher id
        self.results_cp = pd.merge(
            self.results_cp_before,
            self.results_cp_after[["researcher", "div_mean"]],
            on="researcher",
            suffixes=("_before", "_after"),
        )

        # Create the scatter plot
        scatter_plot = (
            alt.Chart(self.results_cp)
            .mark_circle(size=60)
            .encode(
                x=alt.X("div_mean_before:Q", title="Div Mean Before"),
                y=alt.Y("div_mean_after:Q", title="Div Mean After"),
                color=alt.Color("status:N", legend=alt.Legend(title="Status")),
                tooltip=["status:N", "div_mean_before:Q", "div_mean_after:Q"],
            )
            .properties(
                title="Comparison of Div Mean Before and After", width=600, height=400
            )
        )

        # Create the fitted lines
        fitted_lines = scatter_plot.transform_regression(
            "div_mean_before", "div_mean_after", groupby=["status"]
        ).mark_line()

        # Create a 45-degree line
        degree_line = (
            alt.Chart(
                pd.DataFrame(
                    {
                        "x": [
                            self.results_cp["div_mean_before"].min(),
                            self.results_cp["div_mean_before"].max(),
                        ]
                    }
                )
            )
            .mark_line(color="black", strokeDash=[3, 3])
            .encode(x="x", y="x")
        )

        # Combine the scatter plot, the fitted lines, and the 45-degree line
        final_plot = scatter_plot + fitted_lines + degree_line

        # Save the plot as a PNG file
        png_str = vlc.vegalite_to_png(vl_spec=final_plot.to_json(), scale=2)

        with open(
            "eic_case_studies/data/06_outputs/consortia/06_scatterplot_div_mean_before_after.png",
            "wb",
        ) as f:
            f.write(png_str)

        self.next(self.end)

    @step
    def end(self):
        """
        End the flow.
        """
        pass


def create_whisker_boxplot(order, label, results):
    # filter results by label in publication_year
    if label != "all":
        results = results[results["publication_year"] == label]

    chart = (
        alt.Chart(results)
        .mark_boxplot()
        .encode(
            x=alt.X("proposal_call_id:N", sort=order),
            y="div:Q",
            color=alt.Color("contains_open:N", legend=alt.Legend(title="Open Call")),
        )
        .properties(width=800, height=400)
    )

    # Create a box plot for 'variety'
    chart_variety = (
        alt.Chart(results)
        .mark_boxplot(size=6)
        .encode(
            x=alt.X("proposal_call_id:N", sort=order, axis=None),
            y="variety:Q",
            color=alt.Color("contains_open:N", legend=alt.Legend(title="Open Call")),
        )
        .properties(width=270, height=170)
    )

    # Create a box plot for 'balance'
    chart_balance = (
        alt.Chart(results)
        .mark_boxplot(size=6)
        .encode(
            x=alt.X("proposal_call_id:N", sort=order, axis=None),
            y="balance:Q",
            color=alt.Color("contains_open:N", legend=alt.Legend(title="Open Call")),
        )
        .properties(width=270, height=170)
    )

    # Create a box plot for 'average_disparity'
    chart_average_disparity = (
        alt.Chart(results)
        .mark_boxplot(size=6)
        .encode(
            x=alt.X("proposal_call_id:N", sort=order, axis=None),
            y="average_disparity:Q",
            color=alt.Color("contains_open:N", legend=alt.Legend(title="Open Call")),
        )
        .properties(width=270, height=170)
    )

    # Concatenate the three charts horizontally
    combined_chart = alt.vconcat(chart_variety, chart_balance, chart_average_disparity)

    # Concatenate the combined chart with the main one vertically
    final_chart = alt.hconcat(chart, combined_chart)

    # add a title
    final_chart = final_chart.properties(
        title=f"Whisker Box Plot for Disparity Components - {label}"
    )

    png_str = vlc.vegalite_to_png(vl_spec=final_chart.to_json(), scale=2)

    with open(
        f"eic_case_studies/data/06_outputs/consortia/01_whisker_box_plot_{label}.png",
        "wb",
    ) as f:
        f.write(png_str)


def density_plot(results, mean_values, label):

    # subset results by label in publication_year
    results = results[results["publication_year"] == label]
    mean_values = mean_values[mean_values["publication_year"] == label]

    # Create a density plot for 'div' with mean line
    chart = (
        alt.Chart(results)
        .transform_density("div", as_=["div", "density"], groupby=["status"], extent=[0, 0.005])
        .mark_area(opacity=0.5)
        .encode(
            x="div:Q",
            y="density:Q",
            color=alt.Color("status:N"),
        )
        .properties(width=800, height=400)
    )
    mean_line_div = (
        alt.Chart(mean_values)
        .mark_rule()
        .encode(
            x="div:Q",
            color=alt.Color(
                "status:N",
            ),
            size=alt.value(1),
        )
    )
    chart += mean_line_div

    # Create a density plot for 'variety' with mean line
    chart_variety = (
        alt.Chart(results)
        .transform_density("variety", as_=["variety", "density"], groupby=["status"], extent=[0, 0.08])
        .mark_area(opacity=0.5)
        .encode(
            x="variety:Q",
            y="density:Q",
            color=alt.Color("status:N"),
        )
        .properties(width=270, height=100)
    )
    mean_line_variety = (
        alt.Chart(mean_values)
        .mark_rule()
        .encode(
            x="variety:Q",
            color=alt.Color(
                "status:N",
            ),
            size=alt.value(1),
        )
    )
    chart_variety += mean_line_variety

    # Create a density plot for 'balance' with mean line
    chart_balance = (
        alt.Chart(results)
        .transform_density("balance", as_=["balance", "density"], groupby=["status"])
        .mark_area(opacity=0.5)
        .encode(
            x="balance:Q",
            y="density:Q",
            color=alt.Color("status:N"),
        )
        .properties(width=270, height=100)
    )
    mean_line_balance = (
        alt.Chart(mean_values)
        .mark_rule()
        .encode(
            x="balance:Q",
            color=alt.Color(
                "status:N",
            ),
            size=alt.value(1),
        )
    )
    chart_balance += mean_line_balance

    # Create a density plot for 'average_disparity' with mean line
    chart_average_disparity = (
        alt.Chart(results)
        .transform_density(
            "average_disparity",
            as_=["average_disparity", "density"],
            groupby=["status"],
        )
        .mark_area(opacity=0.5)
        .encode(
            x="average_disparity:Q",
            y="density:Q",
            color=alt.Color("status:N"),
        )
        .properties(width=270, height=100)
    )
    mean_line_average_disparity = (
        alt.Chart(mean_values)
        .mark_rule()
        .encode(
            x="average_disparity:Q",
            color=alt.Color(
                "status:N",
            ),
            size=alt.value(1),
        )
    )
    chart_average_disparity += mean_line_average_disparity

    # Concatenate the three charts horizontally
    combined_chart = alt.vconcat(chart_variety, chart_balance, chart_average_disparity)

    # Concatenate the combined chart with the main one vertically
    final_chart = alt.hconcat(chart, combined_chart)

    # add a title
    final_chart = final_chart.properties(
        title=f"Density Plot for Disparity Components - {label}"
    )

    png_str = vlc.vegalite_to_png(vl_spec=final_chart.to_json(), scale=2)

    with open(
        f"eic_case_studies/data/06_outputs/consortia/02_density_plot_{label}.png",
        "wb",
    ) as f:
        f.write(png_str)


def create_wbox_status(results, label):
    # Create a box plot for each researcher_last_evaluation_status
    chart = (
        alt.Chart(results)
        .mark_boxplot()
        .encode(
            x=alt.X("status:N"),
            y="div:Q",
        )
        .properties(width=800, height=400)
    )
    # Create a box plot for 'variety'
    chart_variety = (
        alt.Chart(results)
        .mark_boxplot(size=6)
        .encode(
            x=alt.X("status:N", axis=None),
            y="variety:Q",
        )
        .properties(width=270, height=125)
    )

    # Create a box plot for 'balance'
    chart_balance = (
        alt.Chart(results)
        .mark_boxplot(size=6)
        .encode(
            x=alt.X("status:N", axis=None),
            y="balance:Q",
        )
        .properties(width=270, height=125)
    )

    # Create a box plot for 'average_disparity'
    chart_average_disparity = (
        alt.Chart(results)
        .mark_boxplot(size=6)
        .encode(
            x=alt.X("status:N", axis=None),
            y="average_disparity:Q",
        )
        .properties(width=270, height=125)
    )

    # Concatenate the three charts horizontally
    combined_chart = alt.vconcat(chart_variety, chart_balance, chart_average_disparity)

    # Concatenate the combined chart with the main one vertically
    final_chart = alt.hconcat(chart, combined_chart)

    # Add title
    final_chart = final_chart.properties(
        title=f"Whisker Box Plot for Disparity Components by Researcher Status - {label}"
    )

    png_str = vlc.vegalite_to_png(vl_spec=final_chart.to_json(), scale=2)

    with open(
        f"eic_case_studies/data/06_outputs/consortia/03_whisker_box_status_plot_{label}.png",
        "wb",
    ) as f:
        f.write(png_str)


if __name__ == "__main__":
    ResearcherConsortiaPlots()
