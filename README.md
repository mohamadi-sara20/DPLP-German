# DPLP RST Parser for German #

## 1. Introduction

This repository is a fork of [DPLP Parser](https://github.com/jiyfeng/DPLP) that customizes the code for German language by adding new pieces of codes and other resources. A ready to use model is trained by the Postdam Universities RST Corpus [TODO: correct the names and add links]. The pretrained model can be found in **./data/de** along with all its belongings. The source code consists of several python scripts in the root directory, prefixed by _ger_.

## 2. Runtime Docker Image ##

DPLP Parser was written Python 2 that is discontinued, and depends on libraries that can't be installed by the package managers anymore. Therefore a prebuild Docker image (_mohamadisara20/dplp-env_) is created and shared in Docker Hub that contains all of them preinstalled. Both German and the original English DPLP parsers can be executed within this image. The image can be found here:
https://hub.docker.com/repository/docker/mohamadisara20/dplp-env/general
There are two tags created for this docker image:

- **latest**: the default tag suitable for running the English DPLP parser.
- **ger**: the tag created for running German DPLP. It has some language-specific libraries and data files pre-installed.

## 3. RST Parsing from text files ##

The German RST parser's main script is ger_predict_dis_from_txt.py (it generates RST trees in both .dis and .rs3 formats). All the preprocessings, segmentation and parsing are included; so the script simply takes text files as input and generates the RST trees as its ultimate output. You can follow these steps to parse a batch of text files using this script in a docker container:

1- in terminal, change directory to the repo's root path:
```
cd path_to_rst_german
```

2- copy the _.txt_ input files in a single subdirectory in **./data** (let's say: **data/input**). Please don't choose any place ourside the current directory.

3- run the following command:
```
docker run -d -v $(pwd):/home/DPLP -w /home/DPLP mohamadisara20/dplp-env:ger python3 ger_predict_dis_from_txt.py data/input
```
- **Important**: replace `data/input` with the _relative_ path to your input text files. You don't need to change anything else for standard parsing. For customization with additional args, refer to the parser script's source code.


4- There are several ways to check the progress or debug errors:

- verify the new files created in the input path
- replace `-d` option with `-it` to get all logs printed throughout the process
- use `docke logs` to see the logs generated by the Docker container

5- you will see new files created throughout the parsing process. When _.rs3_ files are generated, it means that the process is finished.

## 4. RST Parsing with the Restful API ##

The script **ger_rest_api.py** creates a REST-API server that allows us to perform RST parsing remotely or integrate it into a web application. It can be launched by running:

```
docker run -d -p 5000:5000 -v $(pwd):/home/DPLP -w /home/DPLP mohamadisara20/dplp-env python3 ger_rest_api.py
```

Then the REST-API will listen to port 5000 and accept JSON requests from the subpath **dplp** and return the RST tree in two formats _dis_ and _rs3_. You can test it using this command:

```
curl -d '{"text":"Ich bin gut."}' -H 'Content-Type: application/json' http://127.0.0.1:5000/dplp
```

You will get a result like this:

```
{
    "dis": "(Root (leaf 1) (rel2par None) (text _!lorem ipsum_!))\n",
    "rs3":"<rst>\n
        <header>\n <relations>\n      <rel name=\"Antithesis\" type=\"rst\"/>\n      
            <rel         name=\"Background\" type=\"rst\"/>\n      <rel name=\"Cause\" type=\"rst\"/>\n      <rel name=\"Circumstance\" type=\"rst\"/>\n      <rel name=\"Concession\" type=\"rst\"/>\n      <rel name=\"Condition\" type=\"rst\"/>\n      <rel name=\"Conjunction\" type=\"multinuc\"/>\n      <rel name=\"Contrast\" type=\"multinuc\"/>\n      <rel name=\"Disjunction\" type=\"multinuc\"/>\n      <rel name=\"Elaboration\" type=\"rst\"/>\n      <rel name=\"Enablement\" type=\"rst\"/>\n      <rel name=\"Evaluation\" type=\"rst\"/>\n      <rel name=\"Evidence\" type=\"rst\"/>\n      <rel name=\"Interpretation\" type=\"rst\"/>\n      <rel name=\"Joint\" type=\"multinuc\"/>\n      <rel name=\"Justify\" type=\"rst\"/>\n      <rel name=\"Motivation\" type=\"rst\"/>\n      <rel name=\"Otherwise\" type=\"rst\"/>\n      <rel name=\"Preparation\" type=\"rst\"/>\n      <rel name=\"Purpose\" type=\"rst\"/>\n      <rel name=\"Restatement\" type=\"rst\"/>\n      <rel name=\"Result\" type=\"rst\"/>\n      <rel name=\"Sequence\" type=\"multinuc\"/>\n      <rel name=\"Solutionhood\" type=\"rst\"/>\n      <rel name=\"Summary\" type=\"rst\"/>\n
        </relations>\n</header>\n  
        <body>\n
            <segment id=\"2\">Ich bin gut .</segment>\n 
        </body>\n </rst>",
    "dis_url":"rstout/5d74b5b2-18c6-4443-9d5f-b67ebfe947a3/document.dis",
    "rs3_url":"rstout/5d74b5b2-18c6-4443-9d5f-b67ebfe947a3/document.rs3","uid":"5d74b5b2-18c6-4443-9d5f-b67ebfe947a3"}
```

## 5. Training Your Own RST Parser ##
You can train your own parser using a corpus of RST trees. German parser uses _.rs3_ files for training. Training consists of these steps:

1- Divide the corpus into train, dev and test set and save them in three subdirectores **training/**, **dev/** and **test/** in a _base_ directory ; for instance, if the base directory is _data/base_dir_, we will get these sub directories: **data/base_dir/training**, **data/base_dir/dev** and **data/base_dir/test**.

2- Review the content of the relation mapping file: _parsing\_eval\_metrics/rel\_mapping.json_. It should contain all relations used in the whole corpus (train, dev and test set); otherwise training can fail and show undefind bahaviors.

3- The script _ger\_train\_parser.py_ tringgers the trining process on the base directory. You can use the following command to run the training code in the docker container:

```
docker run -d -v $(pwd):/home/DPLP -w /home/DPLP mohamadisara20/dplp-env:ger python3  ger_train.py data/base_dir
```

4- The model will be saved in the file **model/model.pickle.gz** in the base path (e.g. _model/de_)
5- The parser precision scrores will be reported in the file named **results.txt** in the base directory.

## Reference ##

Please read the following paper for more technical details

- Yangfeng Ji, Jacob Eisenstein. *[Representation Learning for Text-level Parsing](http://jiyfeng.github.io/papers/ji-acl-2014.pdf)*. ACL 2014
- Joty, S., Carenini, G., & Ng, R. T. (n.d.). CODRA: A Novel Discriminative Framework for Rhetorical Analysis.
- Shahmohammadi, S., & Stede, M. (2024). Discourse Parsing for German with new RST Corpora. Workshop Proceedings of the 20th Edition of the KONVENS Conference. 
