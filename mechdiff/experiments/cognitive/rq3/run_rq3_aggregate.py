from mechdiff.experiments.cultural.rq3.run_rq3_aggregate import main as cultural_main  # noqa: F401


def main():
    # Delegate to the cultural aggregator; it parses current sys.argv.
    cultural_main()


if __name__ == "__main__":
    main()
