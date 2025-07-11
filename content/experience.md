---
title: 'Experience'
date: 2023-10-24
type: landing

design:
  spacing: '5rem'

# Note: `username` refers to the user's folder name in `content/authors/`

# Page sections
sections:
  - block: resume-experience
    content:
      username: admin
    design:
      # Hugo date format
      date_format: 'January 2006'
      # Education or Experience section first?
      is_education_first: false
  - block: resume-skills
    content:
      title: Technical Skills
      username: admin
      items:
        - name: Python
          description: ''
          percent: 80
          icon: code-bracket
        - name: C++
          description: ''
          percent: 70
          icon: code-bracket
        - name: LaTeX
          description: ''
          percent: 70
          icon: code-bracket
    design:
      show_skill_percentage: true
      align: "center"
  - block: resume-skills
    content:
      title: Hobbies
      username: admin
      items:
        - name: Hiking
          description: 'Hiking really ease me a lot~'
          percent: 70
          icon: person-simple-walk
        - name: Open Source Projects
          description: ''
          percent: 90
          icon: code-bracket
        - name: Volleyball
          description: 'Looking up into the sky~'
          percent: 90
          icon: volleyball
    design:
      show_skill_percentage: true
      align: "center"
  - block: resume-languages
    content:
      title: Languages
      username: admin
---