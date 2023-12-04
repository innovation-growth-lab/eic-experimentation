"""
python eic_benchmark_report_generator/code/benchmark_report.py --highlight EIC
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from eic_benchmark_report_generator import PROJECT_DIR


class RadarPlot:
    """Class to generate a radar/spider plot based on agency scores for each category."""

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.categories = self.df["Category"].unique()
        self.agencies = self.df.columns[2:]

    def _get_sums(self) -> pd.DataFrame:
        """Get summed scores for each category for each agency."""
        return self.df.groupby("Category").sum(numeric_only=True)

    def plot(self, highlight: str = None, disclose: bool = False) -> None:
        """Generate the radar plot with an optional highlight for a specific agency."""

        # Define colors for each agency
        colors = {
            "EIC": "#092640",
            "Innosuisse": "#1F5DAD",
            "Innoviris": "#FF5836",
            "IUK": "#FAB61B",
            "CDTI": "#00B2A2",
            "SIEA": "#ff7f0e",
            "Wallonie": "#d62728",
        }

        sums = self._get_sums()
        labels = list(sums.index)
        num_vars = len(labels)

        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop

        _, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))

        # Hide the default polar grid and tick labels
        ax.axis("off")

        # Manually draw the axis lines for each dimension
        for angle in angles:
            ax.plot([angle, angle], [0, 9], color="gray", linestyle="--", linewidth=1)

        for agency in self.agencies:
            values = sums[agency].tolist()
            values += values[:1]  # Close the loop

            if highlight and agency == highlight:
                alpha_val = 0.8
                line_val = 6
                marker_val = 8
            else:
                alpha_val = 0.4  # Decreased alpha for lines
                line_val = 3
                marker_val = 6

            ax.plot(
                angles,
                values,
                color=colors[agency],
                linewidth=line_val,
                alpha=alpha_val,
                label=(agency if (disclose or agency == highlight) else ""),
                marker="s",
                markersize=marker_val,
            )
            ax.fill(
                angles, values, color=colors[agency], alpha=0.05
            )  # Fixed alpha for fill

        # Adjust the position of the x-axis labels to prevent overlap
        for label, angle in zip(labels, angles[:-1]):
            if angle in (0, np.pi):
                ha, distance = "center", 1.1
            elif 0 < angle < np.pi:
                ha, distance = "left", 1.1
            else:
                ha, distance = "right", 1.1
            ax.text(
                angle,
                8.5 * distance,
                label,
                size=10,
                horizontalalignment=ha,
                verticalalignment="center",
            )

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(180 / num_vars)

        ax.set_yticks([3, 6, 9])
        ax.set_ylim(0, 9)

        # Legend placement at the bottom
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=len(self.agencies),
            handlelength=1,
        )

        plt.tight_layout()
        radar_output_path = os.path.join(
            PROJECT_DIR,
            "eic_benchmark_report_generator/outputs/figures",
            f"radar_plot_{'all' if not highlight else highlight}.png",
        )
        plt.savefig(radar_output_path, dpi=300)
        plt.close()


class HistogramPlot:
    """Class to generate histograms for sub-categories based on agency scores."""

    def __init__(self, dataframe):
        self.df = dataframe
        self.categories = self.df["Category"].unique()
        self.agencies = self.df.columns[2:]

        # Color definition
        self.colors = {
            "EIC": "#092640",
            "Innosuisse": "#1F5DAD",
            "Innoviris": "#FF5836",
            "IUK": "#FAB61B",
            "CDTI": "#00B2A2",
            "SIEA": "#ff7f0e",
            "Wallonie": "#d62728",
        }

    def plot(self, highlight: str = None, disclose: bool = False):
        """Generate the histogram plots with an optional highlight for a specific agency."""
        for i, cat in enumerate(self.categories):
            sub_df = self.df[self.df["Category"] == cat]
            sub_categories = sub_df["Sub-category"].values
            n_subcats = len(sub_categories)

            _, axs = plt.subplots(1, n_subcats, figsize=(20, 5), sharey=True)

            for ax, sub_cat in zip(axs, sub_categories):
                scores = sub_df[sub_df["Sub-category"] == sub_cat][
                    self.agencies
                ].values[0]

                # Add soft gray horizontal grid lines
                ax.yaxis.grid(zorder=0, linestyle="--", color="lightgray", alpha=0.7)

                # Adjust the bar color for highlight using the colors dictionary
                ax.bar(
                    self.agencies,
                    scores,
                    color=[
                        self.colors[agency] if agency == highlight else "gray"
                        for agency in self.agencies
                    ],
                    zorder=2,
                )

                ax.set_title(sub_cat)

                if sub_cat in ["Pipeline", "Forward"]:
                    ax.set_ylim(0, 4.5)
                else:
                    ax.set_ylim(0, 3)
                # else:
                #     ax.set_ylim(0, 3)
                # Removing top and right spines
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                # Adjust the x-axis labels if disclose is True
                if not disclose:
                    ax.set_xticks(
                        [
                            highlight if agency == highlight else ""
                            for agency in self.agencies
                        ]
                    )
                else:
                    ax.set_xticks(self.agencies)

            # plot and save
            plt.tight_layout()
            histogram_output_path = os.path.join(
                PROJECT_DIR,
                "eic_benchmark_report_generator/outputs/figures",
                f"histogram_plot_{'all' if not highlight else highlight}_{i}.png",
            )
            plt.savefig(histogram_output_path, dpi=300)
            plt.close()


def generate_colored_table(df: pd.DataFrame):
    """Generate a colored table from the input data."""

    # Add a total row
    total_row = df[df.columns[2:]].sum(numeric_only=True)
    total_row["Category"] = "Total"
    total_row["Sub-category"] = ""
    df = df.append(total_row, ignore_index=True)

    # Set up the figure and axes
    _, ax = plt.subplots(figsize=(10, 10))

    # Remove the axes
    ax.axis("off")

    # Define a color mapping function
    def color_mapping(val):
        # Check if the value is numeric and apply the color logic
        if pd.api.types.is_number(val):
            if val < 3:
                return "red"
            elif 3 <= val < 6:
                return "yellow"
            elif 6 <= val:
                return "green"
        # Return white color for non-numeric cells
        return "white"

    # Apply the color mapping to the dataframe
    colors = df.applymap(color_mapping)

    # Display the table
    ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellColours=colors.values,
        loc="center",
    )

    output_path = os.path.join(
        PROJECT_DIR,
        "eic_benchmark_report_generator/outputs/figures",
        "colored_table.png",
    )
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_category_histograms(
    dataframe: pd.DataFrame, highlight: str = None, disclose: bool = False
) -> None:
    """Generate histograms for each category based on agency summed scores."""

    # Sum scores for each category for each agency
    summed_scores = dataframe.groupby("Category", sort=False).sum(numeric_only=True)

    # Categories and Agencies
    categories = summed_scores.index
    agencies = dataframe.columns[2:]

    # Color definition for each agency
    colors = {
        "EIC": "#092640",
        "Innosuisse": "#1F5DAD",
        "Innoviris": "#FF5836",
        "IUK": "#FAB61B",
        "CDTI": "#00B2A2",
        "SIEA": "#ff7f0e",
        "Wallonie": "#d62728",
    }

    # Setting up the figure and subplots
    _, axs = plt.subplots(nrows=len(categories), figsize=(12, 15))

    for idx, cat in enumerate(categories):
        scores = summed_scores.loc[cat, agencies]

        # Create the bar chart for the category
        axs[idx].bar(agencies, scores, color=[colors[agency] for agency in agencies])

        # Additional formatting
        axs[idx].set_title(f"Category: {cat}")
        axs[idx].yaxis.grid(zorder=0, linestyle="--", color="lightgray", alpha=0.7)
        axs[idx].set_ylim(
            0, 9
        )  # Setting y limit to be slightly more than max score for aesthetics
        axs[idx].spines["top"].set_visible(False)
        axs[idx].spines["right"].set_visible(False)

        # Adjust the x-axis labels if disclose is True
        axs[idx].set_xticks(range(len(agencies)))
        if disclose:
            axs[idx].set_xticklabels(agencies)
        else:
            axs[idx].set_xticklabels(
                [agency if agency == highlight else "" for agency in agencies]
            )

    # plot and save
    plt.tight_layout()
    if highlight:
        histogram_output_path = os.path.join(
            PROJECT_DIR,
            "eic_benchmark_report_generator/outputs/figures",
            f"histogram_plot_{highlight}_main.png",
        )
    else:
        histogram_output_path = os.path.join(
            PROJECT_DIR,
            "eeic_benchmark_report_generatoric/outputs/figures",
            "histogram_plot_all_main.png",
        )

    plt.savefig(histogram_output_path, dpi=300)
    plt.close()


def preprocess_data(data_path: str) -> pd.DataFrame:
    """Load, preprocess, and return the benchmarking results data."""

    # Read data from the specified path
    df = pd.read_csv(data_path, sep=",")

    # Rename column Wallonie Entreprendre as Wallonie
    df = df.rename(columns={"Wallonie Entreprendre": "Wallonie"})

    # Sum rows corresponding to Extensive and Intensive margin
    # Call new sub-category "Data Use"
    orig_rows = df.loc[
        (df["Sub-category"] == "Extensive margin")
        | (df["Sub-category"] == "Intensive margin")
    ]
    new_row = orig_rows.sum(numeric_only=True)
    new_row["Sub-category"] = "Data Use"
    new_row["Category"] = "Data Collection & Integration"
    df = pd.concat(
        [df, new_row.to_frame().T.set_index(pd.Index([5]))], ignore_index=False
    ).sort_index()

    # Remove orig_rows
    df = df.drop(orig_rows.index).reset_index(drop=True)

    # Convert any numeric columns into floats
    df[df.columns[2:]] = df[df.columns[2:]].astype(float)

    return df


def main():
    """Main function to execute the script."""
    # Argument parsing
    parser = argparse.ArgumentParser(description="Generate benchmarking plots.")
    parser.add_argument(
        "--highlight", type=str, help="Agency to highlight in the plots."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=f"{PROJECT_DIR}/eic_benchmark_report_generator/data/benchmarking_results.txt",
        help="Path to the data file.",
    )
    parser.add_argument(
        "--disclose",
        action="store_true",
        help="Whether to disclose the agency name in the plots.",
    )
    parser.add_argument(
        "--all", action="store_true", help="Iterate over each agency, highlighting it."
    )
    args = parser.parse_args()

    # assert disclose is only False when highlight is not None, unless --all is used
    if not args.disclose and not args.all:
        assert (
            args.highlight is not None
        ), "Highlight must be None when disclose is False."

    # log that the use of `all` will override other arguments
    if args.all:
        print("`all` argument used, overriding other arguments.")

    # Load and preprocess data
    df = preprocess_data(args.data_path)

    # Ensure output directory exists
    output_dir = f"{PROJECT_DIR}/eic_benchmark_report_generator/outputs/figures"
    os.makedirs(output_dir, exist_ok=True)

    agencies = df.columns[2:].tolist()

    if args.all:
        for agency in agencies:
            # Radar plot
            rp = RadarPlot(df)
            rp.plot(highlight=agency, disclose=False)

            # Histogram plot
            hp = HistogramPlot(df)
            hp.plot(highlight=agency, disclose=False)

            # category histogram plot
            plot_category_histograms(df, highlight=agency, disclose=False)
    else:
        # Radar plot
        rp = RadarPlot(df)
        rp.plot(highlight=args.highlight, disclose=args.disclose)

        # Histogram plot
        hp = HistogramPlot(df)
        hp.plot(highlight=args.highlight, disclose=args.disclose)

        # category histogram plot
        plot_category_histograms(df, highlight=args.highlight, disclose=args.disclose)

    # output table (not used)
    # generate_colored_table(df)


if __name__ == "__main__":
    main()
