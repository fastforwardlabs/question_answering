# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2021
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ###########################################################################

# helper functions for downloading various QA datasets
import urllib.request

from qa.utils import absolute_path, create_path
from qa.utils import absolute_path, create_path


def download_squad(version=2):
    if version not in [1, 2]:
        print("Please specificy SQuAD version number.")
        return

    url_base = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
    data_dir = "data/squad/"

    if version == 1:
        print("Downloading SQuAD1.1 training and development sets...")
        train = "train-v1.1.json"
        dev = "dev-v1.1.json"

    elif version == 2:
        print("Downloading SQuAD2.0 training and development sets...")
        train = "train-v2.0.json"
        dev = "dev-v2.0.json"

    create_path(absolute_path(data_dir, train))
    create_path(absolute_path(data_dir, dev))

    urllib.request.urlretrieve(url_base + train, absolute_path(data_dir, train))
    urllib.request.urlretrieve(url_base + dev, absolute_path(data_dir, dev))
    return


def download_covidQA():
    print("Downloading COVID-QA dataset...")

    url = "https://github.com/deepset-ai/COVID-QA/raw/master/data/question-answering/COVID-QA.json"
    output_filename = absolute_path("data/medical/COVID-QA.json")

    create_path(output_filename)
    urllib.request.urlretrieve(url, output_filename)
    return


def main():
    GET_SQUAD = True
    GET_COVID = False

    if GET_SQUAD:
        download_squad(version=2)
    if GET_COVID:
        download_covidQA()


if __name__ == "__main__":
    main()
