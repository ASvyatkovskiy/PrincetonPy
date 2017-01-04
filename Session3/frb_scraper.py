#!/usr/bin/env python
from pyvirtualdisplay import Display

from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By

import time

import pdb

def get_all_items(max_num_items=None):
    #list to store alll scraped data
    all_items = list()

    #Display - read about pyvirtualdisplay
    display = Display(visible=0, size=(1024, 768))
    display.start()
    #webdriver - read about selenium.webdriver
    driver = webdriver.Firefox()
    
    #this is a starting page we are scraping
    driver.get("http://www.federalreserve.gov/apps/reportforms/default.aspx")
    #Every element on the HTML page can be located using CSS selectors.
    #Opening the starting page in Chrome, right click on the drop-down menu, click "Inspect" we see a tag on the right highlighted, we copy it's id - MainContent_ddl_ReportForms
    #Knowing the id of dropdown menu, we can locate it with Selenium like this
    main_menu = WebDriverWait(driver,10).until(EC.presence_of_element_located((By.CSS_SELECTOR,"#MainContent_ddl_ReportForms")))
    #Drop down menu is an HTML table of options which can be verified in Chrome browser (Developer Tools, that pop up when you right click and press "Inspect" on an element)
    #Following returns all of the options - rows in that table
    form_options = main_menu.find_elements_by_tag_name("option")
    #We count them
    option_count = len(form_options)
    if max_num_items is not None:
        option_count = min(max_num_items,option_count) 
        if option_count < 2: 
            print "Need to inspect at least one option"
            exit(1)
    #Next, we loop over all of them - essentially like we scrolling down the drop down menu and clicking on each every form 
    for form_i in xrange(1,option_count):
        #Get web element corresponding to a form
        form = form_options[form_i]
        #Click as a mouse click-action in browser 
        form.click()
        #Get text, because we need to store the form number
        form_id = form.text
        #Locate a web element corresponding to the submit button. By CSS selector which we found by inspection in Chrome browser (same logic as above)
        submit_button = WebDriverWait(driver,3).until(EC.presence_of_element_located((By.CSS_SELECTOR,"#MainContent_btn_GetForm")))
        #Click as a mouse click-action in browser 
        submit_button.click()      
        #Prepare data structures to store all the info we want to scrape
        a = dict.fromkeys(['Description','OMB','Background','RespondentPanel','Frequency','PublicRelease'])
        #We are on a web page after submit-click, following will search for all items of interest. Or for corresponding
        #web-elements 
        for el in a.keys():
            try:
                item = driver.find_element_by_css_selector("#MainContent_lbl_"+el+"_data") 
		#item = WebDriverWait(driver,5).until(EC.presence_of_element_located((By.CSS_SELECTOR,"#MainContent_lbl_"+el+"_data")))
                #Once found it will store them in our dictionary, if not it will proceed to "except" section and do nothing
                a[el] = item.text 
            except: 
                #case when there is no such field
                pass
        #we need form number as well
        a['FormNumber'] = form_id
        #keeping them all in one list, which will have a dictionary per Form Number - and later, a row in your excel file per Form number
        all_items.append(a)
    
        #Ok, that part bothers me a little: it looks like I have to refresh "form_options" each time... 
        #Otherwise I get following exception: selenium.common.exceptions.StaleElementReferenceException: Message: Element not found in the cache - perhaps the page has changed since it was looked up
        driver.get("http://www.federalreserve.gov/apps/reportforms/default.aspx")
        main_menu = WebDriverWait(driver,10).until(EC.presence_of_element_located((By.CSS_SELECTOR,"#MainContent_ddl_ReportForms")))
        form_options = main_menu.find_elements_by_tag_name("option")

    driver.close()
    display.stop()

    return all_items

def main():
    pdb.set_trace()
    all_items = get_all_items(2)
    #print all_items

    #Convert our data structre and write to CSV
    import csv
    keys = ['FormNumber','Description','OMB','Background','RespondentPanel','Frequency','PublicRelease']
    with open('forms.csv', 'wb') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(all_items)

if __name__=='__main__':
    main()
