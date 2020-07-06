<template>
  <v-container>
    <div class="large-12 medium-12 small-12 cell">
      <label>
        File
        <input
          type="file"
          id="files"
          ref="files"
          name="files"
          multiple
          v-on:change="handleFileUploads()"
        />
      </label>
      <v-btn class="ma-10" v-on:click="submitFiles()"> Submit Flies </v-btn>

      <v-btn v-on:click="submitCorrections"> Submit Corrections </v-btn>
    </div>

    <v-row>
      <v-col cols="10">
        <vuecanvas
          v-if="show_canvas"
          v-bind:image="current_image"
          v-bind:activation_map="activation_maps[current_image.image_id]"
          v-bind:w="canvas_width"
          v-bind:h="canvas_height"
          @update_activation_map="updateActivationMap"
        ></vuecanvas>
      </v-col>

      <v-col cols="2">
        <idlist v-if="show_thumbnails" v-bind:images="images" @switch_image="loadImage"></idlist>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
import axios from "axios";
import Canvas from "./Canvas";
import IdList from "./IdList";

export default {
  components: {
    vuecanvas: Canvas,
    idlist: IdList
  },
  data() {
    return {
      images: Array,
      current_image: Object,
      activation_maps: Object,
      files: [],
      prev_id: Number,
      show_canvas: false,
      show_thumbnails: false,
      canvas_width: 500,
      canvas_height: 500
    };
  },
  methods: {
    handleFileUploads() {
      this.files = this.$refs.files.files;
    },
    submitFiles() {
      let path = "http://localhost:5000/playground";

      let form_data = new FormData();

      for (let i = 0; i < this.files.length; i++) {
        let file = this.files[i];
        form_data.append(String(i), file);
      }

      let payload = this.files;
      axios
        .post(path, form_data, {
          headers: {
            "Content-Type": "multipart/form-data"
          }
        })
        .then(returnVal => {
          console.log("Files successfully uploaded!");
          this.getUploadedFiles();
        })
        .catch(function() {
          console.log("Failed to upload selected files...");
        });
    },
    getUploadedFiles() {
      let path = "http://localhost:5000/playground";

      axios.get(path).then(res => {
        console.log(res.data);
        this.images = res.data.images;
        this.activation_maps = res.data.activation_maps;

        this.loadImage();

        this.show_canvas = true;
        this.show_thumbnails = true;
      });
    },
    loadImage(image) {
      if (image) {
        this.prev_id = this.current_image.image_id;
        this.current_image = image;
      } else {
        this.current_image = this.images[0];
      }
    },
    updateActivationMap(corrected_map) {
      this.activation_maps[this.prev_id] = corrected_map;
    },
    submitCorrections() {
      let path = "http://localhost:5000/playground_receiver";
      axios
        .post(path, { activation_maps: this.activation_maps })
        .then(() => {}).catch(error => {
            console.log(error)
        });
    }
  }
};
</script>