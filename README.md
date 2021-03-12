# ECE143- Group 22 Project

## Dataset
New York City Airbnb Open Data: https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data

## Deploying
1. Go to the `Releases` tab in GitHub
1. Download the latest release source code tarball
1. Extract the tarball to your local machine
1. Patch file permissions: `*.py` and `*.cgi` have write permissions of 0755 and `.htaccess` has permission of 0664
1. Upload all files to the htdocs folder on the server

### Certificates
Website certificates are maintained at the proxy-level by OEC (our webhost). To update a certificate, email [Office of Engineering Computing](support@eng.ucsd.edu) to open a ticket and they will be able to help.

## Developing 
To run a live-reload version of the website for local development, run `docker-compose up -d` in the `/docker` directory. This will host the website at `http://localhost:5000`. Manual refresh of the page is necessary to see any live changes. Run `docker-compose down` to shutdown the server.

## Architecture
The website is designed  to be a dynamically generated site from template files. All of the html template files are in the `templates` folder and folow the Jinja2 syntax. All of the static served content resides in the `static` folder. Look at the existing structure to see how it is organized. All of the json files are located in `/static/content/` and are used to populate the html templates. Below are some notes on editing those json files.

## Content Specification

### Recent News
The recent news section of the home page uses the `news.json` file for loading the data. It is templated in `home.html`.  The structure is as follows:
```
[
	{
		"title": "The update title here",
		"date": "dd Month yyyy",
		"description": "Text describing the update"
	},
	{...}
]
```

### Research
The research page uses the `research.json` file for loading the data. It is templated in `research.html`. The structure is as follows:
```
[
	{
		"title": "Brain-Machine Interfaces",
		"projects":[
			{
				"title": "Free Behavior Analysis",
				"imref": "/static/content/img/projects/freebe.png",
				"contributors": "Paolo G, Wahab A, Kenny C",
				"keywords": "naturalistic behavior, generalized brain-machine interfaces, epilepsy monitoring",
				"description": "To explore the relationship between neural activity and complex human behaviors, we simultaneously capture aspects of natural human behavior and neural signals from electrodes implanted across cortex. This work pursues two paths - improved quantitative behavioral tracking, and the application of machine learning techniques to extract descriptive neural features."
			},
			{...}
		]
	},
	{...}
]
```

### People
The people page uses the `people.json` file for loading the data. It is templated in `people.html`. There are three groups as listed below and only the contents of these groups are populated. All external URLs must be fully qualified with `http:\\` or equivalent. The structure is as follows:
```
{
	"pi":[
		{
			"name": "Vikash Gilja",
			"imref": "../templates/dynamic/img/people/vgilja.jpg",
			"degree": "Ph.D.",
			"title": "Assistant Professor",
			"content":[
				"Electrical & Computer Engineering",
				"Health Sciences Neurograduate Program",
				"University of California, San Diego"
			]
		},
		{...}
	],
	"current": [
		{
			"name": "First Last",
			"imref": "../templates/dynamic/img/people/picname.jpg",
			"degree": "M.S.",
			"title": "Ph.D. Student, Department",
			"education": [
				"M.S., Department, University, year",
				"B.S., Department, University, year"
			],
			"personalstatement": "First is a person of few words, so we wrote some for him/her",
			"researchtopics": "Topic 1, Topic 2, Topic 3",
			"email": "username@eng.ucsd.edu",
			"linkedin": "<optional: if this tag omitted, no icon will be populated>",
			"personalwebsite": "<optional: if this tag is omitted, no icon will be populated>"
		},
		{...}
	],
	"alumni":[
		{
			"name": "First Last",
			"imref": "../templates/dynamic/img/people/picname.jpg",
			"degree": "M.S.",
			"current": "Currently pursing..."
		},
		{...}
	]
}
```

### Publications
The publications page uses the `publications.json` file for loading the data. It is templated in `publications.html`. New filter items are defined by the title labels. The structure is as follows:
```
[
	{
		"title": "Journals",
		"color": "<css color selected>",
		"tag": "<must be unique without any spaces>",
		"contents": [
			{
				"title": "...",
				"publisher": "...",
				"year": 2016,
				"authors": "Last F, Last F, ...",
				"abstract": "Only the first three lines will be shown, but the entire abstract can be included here",
				"url": "<link to pdf or host>",
				"_date": <numerical value in the format yyyymmdd used to sort journals with the same year>
			},
			{...}
		]
	},
	{...}
]
```

### Funding
The funding section of the Research page uses the `funding.json` file for loading the data. It is templated in `research.html`. The structure is as follows:
```
[
	{
		"name": "Funding source",
		"imref": "/static/content/img/funding/FundingSource.png"
	},
	{...}
]
```

### Collaborators
The collaborators section of the Research page uses the `collaborators.json` file for loading the data. It is templated in `research.html`. The structure is as follows:
```
[
	{
		"name": "Group or department | University",
		"contacts": "First Last, First Last, ...",
		"imref": "../templates/dynamic/img/collaborators/SchoolorGroupLogo.png",
		"url": "http://the.group.edu/lab/"
	},
	{...}
]
```


## Contributing

We welcome contributions and requests via issues/pull requests! Please create an Issue outlining your changes using the approapriate template and any proposed changes in a linked Pull Request. Suggestions will be reviewed and merged in accordingly.
