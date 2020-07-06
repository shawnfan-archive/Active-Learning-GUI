
<template>
  <v-container align-center justify-center>
    <v-row>
      <v-col cols="10">
        <vuecanvas
          v-if="current_image"
          v-bind:image="current_image"
          v-bind:activation_map="activation_maps[current_image.image_id]"
          v-bind:w="canvas_width"
          v-bind:h="canvas_height"
          @update_activation_map="updateActivationMap"
          @submit_corrections="toggleSubmissionDialog"
        ></vuecanvas>
      </v-col>

      <v-col cols="2">
        <thumbnails v-bind:images="images" @switch_image="loadImage"></thumbnails>
      </v-col>

      <v-col cols="12">
        <imgprocessor
          v-if="current_image"
          v-bind:image="current_image.data"
          v-bind:w="canvas_width"
          v-bind:h="canvas_height"
        ></imgprocessor>
      </v-col>
    </v-row>

    <!-- Submission dialog -->
    <v-dialog v-model="dialog" persistent max-width="1000">
      <v-card>
        <v-card-title class="headline">Submit activaion maps and retrain model?</v-card-title>
        <v-container>
          <v-row>
            <v-col cols="12">
              <v-select v-model="model_to_train" :items="models" label="Select a model"></v-select>
            </v-col>
          </v-row>
        </v-container>
        <!-- <v-card-text>Latest model: {{model_name}}</v-card-text> -->
        <v-card-actions>
          <v-btn outlined color="red darken-1" text v-on:click="toggleSubmissionDialog">Cancel</v-btn>

          <v-spacer></v-spacer>

          <v-btn
            outlined
            color="green darken-1"
            text
            v-on:click="onSubmit(true)"
          >Train model from scratch</v-btn>
          <v-btn outlined color="green darken-1" text v-on:click="onSubmit(false)">Retrain model</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Loading dialog -->
    <v-dialog v-model="loading" persistent width="800">
      <v-card color="primary" dark>
        <v-card-title>
          {{ loading_message }}
          <v-progress-linear indeterminate color="white"></v-progress-linear>
        </v-card-title>
      </v-card>
    </v-dialog>
  </v-container>
</template>>

<script>
import axios from "axios";
import Chart from "chart.js";
import { ContentLoader } from "vue-content-loader";
//! No curly brackets when importing components
import Canvas from "./Canvas";
import ThumbnailList from "./ThumbnailList";
import ImageProcessor from "./ImageProcessor";

export default {
  components: {
    vuecanvas: Canvas,
    thumbnails: ThumbnailList,
    imgprocessor: ImageProcessor
  },
  data() {
    return {
      drawer: false,
      current_image: null,
      prev_id: null,
      // current_image: {
      //   image_id: "",
      //   disease: "",
      //   path: ""
      // },
      current_map: null,

      images: [
        // {
        //   image_id: string,
        //   disease: string,
        //   path: string
        // },
      ],
      activation_maps: [
        // {string (id): array(activation map)}
      ],
      canvas_width: 600,
      canvas_height: 500,
      skeleton_loader: true,
      skeleton_thumbnail: [1, 2, 3, 4, 5],
      dialog: false,
      model_name: "",
      loaded: false,
      loading: false,
      loading_message: "Initializing Training...",
      canvas_data: new Array(256).fill(0),
      brightness: 100,
      gamma: 100,
      image_histogram_data: new Array(256).fill(0),
      current_histogram_data: new Array(256).fill(0),
      histogram_labels: new Array(256).fill(0),
      pixel_range: [0, 8000],
      model_to_train: "",
      models: [
        "U-Net Segmentation Model (Danilo Pena)",
        "Random Forest Boosting Model (Shawn Fan)",
        "False-Positive and False Negative Networks (Ivan Coronado)"
      ]
    };
  },
  methods: {
    getActivationMap: function() {
      //return promise
      return axios
        .get("http://localhost:5000/active_learning")
        .then(res => {
          this.images = res.data.images;
          this.model_name = res.data.latest_model;
          this.activation_maps = res.data.activation_maps;

          this.current_image = this.images[0];
          // this.skeleton_loader = false;
        })
        .catch(error => {
          console.error(error);
        });
    },
    updateActivationMap: function(corrected_map) {
      this.activation_maps[this.prev_id] = corrected_map;
    },
    loadImage: function(image) {
      if (image) {
        this.prev_id = this.current_image.image_id;
        this.current_image = image;
      } else {
        this.current_image = this.images[0];
        this.prev_id = this.current_image.image_id;
      }
    },
    updateCurrentHistogramLabels: function() {
      let labels = new Array(256).fill("0");
      let interval = (this.pixel_range[1] - this.pixel_range[0]) / 255;
      for (let i = 0; i < labels.length; i += 1) {
        labels[i] = String(Math.floor(this.pixel_range[0] + i * interval));
      }
      labels[0] = String(this.pixel_range[0]);
      labels[255] = String(this.pixel_range[1]);

      this.current_histogram_chart.data.labels = labels;
    },
    reloadCurrentHistogram: function() {
      this.updateCurrentHistogramData();
      this.current_histogram_chart.update();
    },
    toggleSubmissionDialog: function() {
      this.dialog = !this.dialog;
    },
    saveData: function(payload) {
      const path = "http://localhost:5000/active_learning";
      axios
        .post(path, payload)
        .then(() => {
          this.loading_message = "Training request accepted...";
          // this.getActivationMap()
          //   .then(returnVal => {
          //     this.loadActivationMap();
          //   })
          //   .catch(err => console.log("Axios err: ", err));
        })
        .catch(error => {
          console.log(error);
        });

      this.loading = true;
      this.updateTrainingProgress();
    },
    onSubmit: function(from_scratch) {
      // //Store current activation map
      // this.storeActivationMap();
      //Reset loading message
      this.loading_message = "Submitting training request...";
      //Close submission dialog
      this.dialog = false;
      //Open loading window
      this.loading = true;
      //Submit corrections
      const payload = {
        from_scratch: from_scratch,
        activation_maps: this.activation_maps
      };
      this.saveData(payload);
    },
    updateTrainingProgress: function() {
      setTimeout(() => {
        const path = "http://localhost:5000/training_progress";
        axios
          .get(path)
          .then(res => {
            if (res.data.finished) {
              this.updatePredictionProgress();
            } else {
              console.log(res.data.current_epoch);
              //Update loading message
              this.loading_message =
                "Current Epoch: " +
                String(res.data.current_epoch) +
                "/" +
                String(res.data.total_epochs) +
                " " +
                "Estimated Time Remaining: " +
                String(res.data.time_remaining);

              this.updateTrainingProgress();
            }
          })
          .catch(error => {
            console.error(error);
          });
      }, 30000);
    },
    updatePredictionProgress: function() {
      setTimeout(() => {
        const path = "http://localhost:5000/prediction_progress";
        axios
          .get(path)
          .then(res => {
            if (res.data.finished) {
              //Close loading window if all predictions have been made
              this.loading = false;
              this.getActivationMap();
            } else {
              //Update loading message
              this.loading_message =
                "Making Predictions: " +
                String(res.data.progress) +
                "/" +
                String(res.data.total);
              this.updatePredictionProgress();
            }
          })
          .catch(error => {
            console.error(error);
          });
      }, 30000);
    }
  },
  created() {
    this.getActivationMap()
      .then(returnVal => {
        // this.loadActivationMap();
        // this.createHistogram();
        this.loaded = true;
        this.loadImage();
        console.log(this.activation_maps);
      })
      .catch(err => console.log("Axios err: ", err));
  }
};
</script>

<style scoped>
.canvas {
  position: absolute;
  cursor: none;
}
#draw {
  cursor: none;
  opacity: 0.3;
}
#histo {
  opacity: 0.2;
}
</style>