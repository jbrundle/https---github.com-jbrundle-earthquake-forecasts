def get_circle_master_catalog(Latitude,Longitude,Radius,Magnitude):

    output_file_name = "USGS_%s.circle.master.catalog" % datetime.date.today().strftime("%F")
    output_file = open(output_file_name, "w")
    output_file.close()

    data = {
    
    # Adjust these  -   CA-NV
        "minmagnitude": Magnitude,
        "longitude":Longitude,
        "latitude":Latitude,
        "maxradiuskm":Radius,
        "mindepth": 0,         # Leaving these depth params in leads to missing some earthquake events
        "maxdepth": 1000,


    # Date parameters
#        "starttime": "2070/01/01",
        "starttime": "1970/01/01",
        "endtime": "2110/01/01",


    # Leave these
        "eventtype": "earthquake",
        "format": "csv",
        "orderby": "time-asc",
    }
    
    block_size = 20000
    event_offset = 1
    
    print(data)
    
    #   First do a count of events
    
    url = "https://earthquake.usgs.gov/fdsnws/event/1/count?"
    params = urllib.parse.urlencode(data)
    print(params)
    query_string = url + str(params)
    print(query_string)
    
    response_count = urllib.request.urlopen(query_string)
    
    event_count = response_count.readlines()
    number_events = int(event_count[0])
    
    print('')
    print('Number of Events: ', number_events)
    n_blocks = int(event_count[0])/block_size        #   Number of complete blocks
    print('Number of complete blocks of size ' + str(block_size) + ' =', n_blocks)
    print('')
    
    for i_block in range(0,n_blocks):
    
    #   -------------------------------------------------------------
    
        event_offset = i_block * block_size + 1
    
        data.update({'offset':event_offset})
        data.update({'limit':block_size})
    
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query?"
        params = urllib.parse.urlencode(data)
        query_string = url + str(params)
    
        response = urllib.request.urlopen(query_string)
    
        catalog = response.readlines()
        
    #   -------------------------------------------------------------
    
        write_to_file(output_file_name,catalog)
                
    residual_events = number_events
    if number_events > block_size:
        residual_events = number_events%(n_blocks*block_size)
        
    if residual_events > 0:
        if n_blocks == 0:
            event_offset = 1
        if n_blocks > 0:
            event_offset = n_blocks * block_size + 1
        data.update({'offset':event_offset})
        data.update({'limit':block_size})
    
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query?"
        params = urllib.parse.urlencode(data)
        query_string = url + str(params)
    
        response = urllib.request.urlopen(query_string)
        catalog = response.readlines()
        
        write_to_file(output_file_name,catalog)
    
    return None