import copy
import datetime
import json
import flux, time
from flux.job import JobspecV1
from flux.core.inner import ffi, raw

class Taskset:
    def __init__ (self, tasksetid, resourcetype, iteration):
        self.resourcetype = resourcetype
        self.tasksetid = str(tasksetid)
        self.input = {}
        self.pipelinestages = []
        self.inputsize = 0
        self.endofpipeline = False
        self.complete = False
        self.iteration = str (iteration)

    def print_data(self):
        print ('tasksetid: ', self.tasksetid)
        pipelinestages = []
        for pipelinestage in self.pipelinestages:
            pipelinestages.append(pipelinestage.name)
        print('pipline stages: ', pipelinestages)
        print ('input size: ', self.inputsize)
        print ('input-resource map: ')
        print ('#################')
        for key in self.input.keys():
            mapping = self.input[key]
            print ('resource id:', key)
            print ('count: ', mapping['count'])
            print ('images:')
            images = mapping['images']
            for key1 in images.keys():
                image = images[key1]
                print ('id: ', key1, 'data', image)
                print('-------------------')

    def get_resource_type (self):
        return self.resourcetype

    def set_endofpipeline (self, endofpipeline):
        self.endofpipeline = endofpipeline

    def get_endofpipeline (self):
        return self.endofpipeline

    def get_taskset_len (self):
        return len(self.pipelinestages)

    def add_pipelinestage (self, pipelinestage):
        self.pipelinestages.append (pipelinestage)

    def get_exectimes (self, resource_id):
        if resource_id not in self.input:
            print ('resource missing', resource_id)
            return None

        exec_times = {}

        if 'images' in self.input[resource_id].keys ():
            for image_key in self.input[resource_id]['images'].keys():
                if 'starttime' in self.input[resource_id]['images'][image_key] and 'endtime' in self.input[resource_id]['images'][image_key]:
                    starttime_s = self.input[resource_id]['images'][image_key]['starttime']
                    endtime_s = self.input[resource_id]['images'][image_key]['endtime']
                    starttime = datetime.datetime.strptime(starttime_s, '%Y-%m-%d %H:%M:%S')
                    endtime = datetime.datetime.strptime(endtime_s, '%Y-%m-%d %H:%M:%S')
                    difference = endtime - starttime
                    exec_times[image_key] = difference.total_seconds ()
                else:
                    print ('image', image_key, 'not yet complete')
        return exec_times

    def set_complete (self, resource_id, is_complete):
        if resource_id not in self.input.keys ():
            self.input[resource_id] = {}
        self.input[str(resource_id)]['is_complete'] = is_complete

    def get_complete (self, resource_id):
        return self.input[str(resource_id)]['is_complete']

    def add_input (self, rmanager, imanager, pmanager, resource):

        print ('add_input ():', resource.id, self.iteration, self.tasksetid)

        if self.resourcetype == 'CPU' and resource.cputype == False:
            return
        if self.resourcetype == 'GPU' and resource.gputype == False:
            return

        chunksize = resource.get_chunksize (self.resourcetype, pmanager.encode_pipeline_stages(self.pipelinestages))

        images = imanager.get_images (chunksize)

        if len(images) > 0:
            if resource.id not in self.input.keys ():
                self.input[resource.id] = {}
                self.input[resource.id]['images'] = {}
                self.input[resource.id]['count'] = 0
                self.input[resource.id]['complete'] = 0
                self.input[resource.id]['scheduled'] = 0
                self.input[resource.id]['status'] = 'QUEUED'
                self.set_complete (resource.id, False)
                self.input[resource.id]['chunksize'] = chunksize
                self.input[resource.id]['count'] = len(images)

            for image_id in images.keys():
                self.input[resource.id]['images'][image_id] = {}
                self.input[resource.id]['images'][image_id]['data'] = images[image_id]
                self.input[resource.id]['images'][image_id]['collectfrom'] = resource.id
                self.input[resource.id]['images'][image_id]['status'] = 'QUEUED'
                self.input[resource.id]['images'][image_id]['reallocated'] = False
            self.inputsize += self.input[resource.id]['count']
            return 0
        else:
            self.set_complete (resource.id, True)
            return -1

    def add_input_taskset (self, input_taskset, rmanager, pmanager, resource):
        print ('add_input_taskset ():', resource.id)

        if self.resourcetype == 'CPU' and resource.cputype == False:
            return
        if self.resourcetype == 'GPU' and resource.gputype == False:
            return

        chunksize = resource.get_chunksize(self.resourcetype,
                                           pmanager.encode_pipeline_stages(self.pipelinestages))

        '''
        if resource.id not in self.input.keys():
            chunksize = resource.get_chunksize(self.resourcetype,
                                               pmanager.encode_pipeline_stages(self.pipelinestages))
            self.input[resource.id] = {}
            self.input[resource.id]['count'] = 0
            self.input[resource.id]['images'] = {}
            self.input[resource.id]['status'] = 'QUEUED'
            self.input[resource.id]['complete'] = 0
            self.input[resource.id]['scheduled'] = 0
            self.input[resource.id]['chunksize'] = chunksize
            self.set_complete (resource.id, False)
        '''

        #first collect it from its own resource.id
        total_count = 0

        if resource.id in input_taskset.input.keys ():
            resource_input_images = input_taskset.input[resource.id]

            if 'count' in input_taskset.input[resource.id].keys () and input_taskset.input[resource.id]['count'] > 0:
                index = 0
                image_keys = list (input_taskset.input[resource.id]['images'].keys())
                while index < input_taskset.input[resource.id]['count'] and total_count < chunksize:
                    if input_taskset.input[resource.id]['images'][image_keys[index]]['reallocated'] == False and \
                        input_taskset.input[resource.id]['images'][image_keys[index]]['status'] == 'SUCCESS':

                        if resource.id not in self.input.keys():
                            self.input[resource.id] = {}
                            self.input[resource.id]['count'] = 0
                            self.input[resource.id]['images'] = {}
                            self.input[resource.id]['status'] = 'QUEUED'
                            self.input[resource.id]['complete'] = 0
                            self.input[resource.id]['scheduled'] = 0
                            self.input[resource.id]['chunksize'] = chunksize
                            self.set_complete (resource.id, False)

                        self.input[resource.id]['images'][image_keys[index]] = {}
                        self.input[resource.id]['images'][image_keys[index]]['data'] = input_taskset.input[resource.id]['images'][image_keys[index]]['data']
                        self.input[resource.id]['images'][image_keys[index]]['collectfrom'] = resource.id
                        self.input[resource.id]['images'][image_keys[index]]['status'] = 'QUEUED'
                        self.input[resource.id]['images'][image_keys[index]]['reallocated'] = False
                        input_taskset.input[resource.id]['images'][image_keys[index]]['reallocated'] = True
                        total_count += 1
                    index += 1

        if total_count < chunksize:
            #search for the rest of the images
            resource_keys = list (input_taskset.input.keys ())

            for resource_key in resource_keys:
                if resource_key == resource.id:
                    continue

                resource_input_images = input_taskset.input[resource_key]
                if 'count' in input_taskset.input[resource_key].keys () and input_taskset.input[resource_key]['count'] > 0:
                    index = 0
                    image_keys = list (input_taskset.input[resource_key]['images'].keys())

                    while index < input_taskset.input[resource_key]['count'] and \
                        total_count < chunksize:
                        if input_taskset.input[resource_key]['images'][image_keys[index]]['reallocated'] == False and \
                            input_taskset.input[resource_key]['images'][image_keys[index]]['status'] == 'SUCCESS':

                            if resource.id not in self.input.keys():
                                self.input[resource.id] = {}
                                self.input[resource.id]['count'] = 0
                                self.input[resource.id]['images'] = {}
                                self.input[resource.id]['status'] = 'QUEUED'
                                self.input[resource.id]['complete'] = 0
                                self.input[resource.id]['scheduled'] = 0
                                self.input[resource.id]['chunksize'] = chunksize
                                self.set_complete (resource.id, False)

                            self.input[resource.id]['images'][image_keys[index]] = {}
                            self.input[resource.id]['images'][image_keys[index]]['data'] = input_taskset.input[resource_key]['images'][image_keys[index]]['data'] 
                            self.input[resource.id]['images'][image_keys[index]]['collectfrom'] = resource_key
                            self.input[resource.id]['images'][image_keys[index]]['status'] = 'QUEUED'
                            self.input[resource.id]['images'][image_keys[index]]['reallocated'] = False
                            input_taskset.input[resource_key]['images'][image_keys[index]]['reallocated'] = True
                            total_count += 1
                            self.input[resource.id]['count'] += 1
                            #add flux dependency
                            print ('adding dependency', resource_key, resource.id)
                            h = flux.Flux()
                            r = h.rpc (b"exception.view.get", {"search":"", "level":"job", "jobid_o":"", "jobid_s":str(resource_key), \
                                       "jobid_d":str(resource.id), "endrank_s":-1, "startrank_s":-1, "endrank_d":-1, "startrank_d":-1, \
                                       "viewtype":2, "dependencytype":"UP", "component":"", "componenttype":""})
                            print (r)
                        index += 1

        if total_count == 0:
            print ('add_input_taskset (): could not find any images')
            return -1
        else:
            self.input[resource.id]['count'] = total_count

        return 0

    def submit_support (self, rmanager, pmanager, resource_ids):
        print ('submitting support taskset', self.tasksetid)
        taskset = {}
        taskset['pipelinestages'] = pmanager.encode_pipeline_stages(self.pipelinestages)
        taskset['id'] = self.tasksetid
        taskset['resourcetype'] = self.resourcetype
        taskset['input'] = {}
        taskset['iteration'] = self.iteration
        if len (resource_ids) == 0:
            resource_ids = self.input.keys ()

        for resource_id in resource_ids:
            taskset['input'][str(resource_id)] = {}
            taskset['input'][str(resource_id)]['count'] = self.input[resource_id]['count']
            taskset['input'][str(resource_id)]['images'] = self.input[resource_id]['images']

            self.input[resource_id]['status'] = 'SCHEDULED'
            self.input[resource_id]['scheduled'] = self.input[resource_id]['count']

            images = self.input[resource_id]['images']

            for image_id in images.keys ():
                self.input[resource_id]['images'][image_id]['status'] = 'SCHEDULED'
                self.input[resource_id]['images'][image_id]['scheduledtime'] = str(datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S'))

        f = flux.Flux ()
        r = f.rpc(b"parslmanager.taskset.submit", taskset)

        for resource_id in resource_ids:
            resource = rmanager.get_resource(resource_id)
            resource.set_current_taskset (self.resourcetype, self.iteration, self.tasksetid)
            resource.set_support_taskset (self.iteration, self.tasksetid, self.resourcetype)
            resource.set_current_taskset (self.resourcetype, 'SUPPORT')
        return ""

    def submit_main (self, rmanager, pmanager, resource_ids):
        print ('submitting main taskset', self.tasksetid)
        taskset = {}
        taskset['pipelinestages'] = pmanager.encode_pipeline_stages(self.pipelinestages)
        taskset['id'] = self.tasksetid
        taskset['resourcetype'] = self.resourcetype
        taskset['input'] = {}
        taskset['iteration'] = self.iteration
        if len (resource_ids) == 0:
            resource_ids = self.input.keys ()

        for resource_id in resource_ids:
            taskset['input'][str(resource_id)] = {}
            taskset['input'][str(resource_id)]['count'] = self.input[resource_id]['count']
            taskset['input'][str(resource_id)]['images'] = self.input[resource_id]['images']
            self.input[resource_id]['status'] = 'SCHEDULED'
            self.input[resource_id]['scheduled'] = self.input[resource_id]['count']

            images = self.input[resource_id]['images']
            for image_id in images.keys ():
                self.input[resource_id]['images'][image_id]['status'] = 'SCHEDULED'
                self.input[resource_id]['images'][image_id]['scheduledtime'] = str(datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S'))

        f = flux.Flux ()
        f.rpc(b"parslmanager.taskset.submit", taskset)

        for resource_id in resource_ids:
            resource = rmanager.get_resource(resource_id)
            resource.set_current_taskset (self.resourcetype, self.iteration, self.tasksetid)
            resource.set_main_taskset (self.iteration, self.tasksetid, self.resourcetype)
            resource.set_current_taskset (self.resourcetype, 'MAIN')

        return ""

    def submit (self, rmanager, pmanager, resource_ids):
        print ('submitting taskset', self.tasksetid)
        taskset = {}
        taskset['pipelinestages'] = pmanager.encode_pipeline_stages(self.pipelinestages)
        taskset['id'] = self.tasksetid
        taskset['resourcetype'] = self.resourcetype
        taskset['input'] = {}
        taskset['iteration'] = self.iteration

        if len (resource_ids) == 0:
            resource_ids = self.input.keys ()

        for resource_id in resource_ids:
            taskset['input'][str(resource_id)] = {}
            taskset['input'][str(resource_id)]['count'] = self.input[resource_id]['count']
            taskset['input'][str(resource_id)]['images'] = self.input[resource_id]['images']
            self.input[resource_id]['status'] = 'SCHEDULED'
            self.input[resource_id]['scheduled'] = self.input[resource_id]['count']

            images = self.input[resource_id]['images']
            for image_id in images.keys ():
                self.input[resource_id]['images'][image_id]['status'] = 'SCHEDULED'
                self.input[resource_id]['images'][image_id]['scheduledtime'] = str(datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S'))

        f = flux.Flux ()
        f.rpc(b"parslmanager.taskset.submit", taskset)

        for resource_id in resource_ids:
            resource = rmanager.get_resource(resource_id)
            resource.set_current_taskset (self.resourcetype, self.iteration, self.tasksetid)

        return ""

    def get_status (self, resource_id):
        f = flux.Flux ()

        print ('getting status')

        if self.input[resource_id]['complete'] == self.input[resource_id]['count']:
            print (self.tasksetid, resource_id, 'already complete')
            return

        r = f.rpc("parslmanager.taskset.status", {"tasksetid": self.tasksetid, "workerid":resource_id}).get()

        print (resource_id, self.iteration, self.tasksetid, r)

        report = r['report']

        if type (report) is not dict and report == 'empty':
            print ('empty report')
            return

        for imageid in report.keys ():
            data = report[imageid]
            status = data['status']
            if status == 'SUCCESS':
                self.input[resource_id]['images'][imageid]['status'] = 'SUCCESS'
                self.input[resource_id]['images'][imageid]['starttime'] = data['starttime']
                self.input[resource_id]['images'][imageid]['endtime'] = data['endtime']
            elif status == 'FAILURE':
                self.input[resource_id]['images'][imageid]['status'] = 'FAILURE'
            else:
                print ('unknown status')
            self.input[resource_id]['complete'] += 1
            self.input[resource_id]['scheduled'] -= 1

        print (self.input[resource_id]['complete'], self.input[resource_id]['scheduled'], self.input[resource_id]['count'])

