# DOC defines the version of the document being generated.
# It should be one of: markup, local, submission, final, finaldraft, tr, trdraft
# See paperversions.tex for the explanation of how the versions differ.
#
# Use 'make DOC' for appropriate values of DOC to build the corresponding document

DOC = draft
# Set DOC to one of draft, local, submission, final, report

# Name of the top-level TeX file sans .tex extension
CONF = paper

TARGETS = $(DOC).pdf

SECTIONDIR = sections

TEXS = $(DOC).tex abstract.tex header.tex body.tex $(SECTIONDIR)/*.tex 

STYS = *.sty

# included figures
FIGS = 

# change to wherever commondefs is located
MAKEDIR = make

include $(MAKEDIR)/commondefs

# change to wherever you put paperversions.tex and the .sty files.
TEXDIR = tex-macros

default: $(TARGETS)

BIBFILE = references.bib

$(DOC).pdf: $(TEXS) $(FIGS) $(DOC).stamp $(DOC).bbl $(BIBFILE) $(STYS)
$(DOC).bbl: $(BIBFILE) $(DOC).stamp

$(DOC).tex: $(CONF).tex
	sed s/DOC/$(DOC)/ < $(CONF).tex > $(DOC).tex
   
LDIRT = local.tex draft.tex submission.tex final.tex finaldraft.tex tr.tex trdraft.tex web.tex *.fdb_latexmk *.fls draft.pdf final.pdf submission.pdf

include $(COMMONRULES)
