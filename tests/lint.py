import click
import logging
from pylint.lint import Run


@click.command()
@click.option(
    "--path", help="path to directory you want to run pylint", default="./src", type=str
)
@click.option(
    "--threshold", help="score threshold to fail pylint runner", default=7, type=float
)
def main(path, threshold):

    logging.info(
        "PyLint Starting | " "Path: {} | " "Threshold: {} ".format(path, threshold)
    )

    results = Run([path], do_exit=False)

    final_score = results.linter.stats.global_note

    if final_score < threshold:

        message = (
            "PyLint Failed | "
            "Score: {} | "
            "Threshold: {} ".format(final_score, threshold)
        )

        logging.error(message)
        raise Exception(message)

    else:
        message = (
            "PyLint Passed | "
            "Score: {} | "
            "Threshold: {} ".format(final_score, threshold)
        )

        logging.info(message)

        exit(0)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
