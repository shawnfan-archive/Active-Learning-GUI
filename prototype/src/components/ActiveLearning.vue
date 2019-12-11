<template>
  <v-app id="main">
    <v-container align-center justify-center>
      <v-toolbar dense class="my-8">
        <v-toolbar-title>Image ID: {{ current_image.image_id }} Disease: {{ current_image.disease }}</v-toolbar-title>

        <v-spacer></v-spacer>

        <v-toolbar-items>
          <v-btn text v-on:click="onSubmit">Save and Retrain Model</v-btn>
          <v-btn text v-on:click="loadActivationMap">Load Activation Map</v-btn>
        </v-toolbar-items>
      </v-toolbar>

      <!-- <div id="container"> -->
      <v-row align="center" justify="center">
        <h2>Canvas: {{canvas_width}} by {{canvas_height}}</h2>
      </v-row>

      <v-row align="center" justify="center">
        <v-col>
          <!-- <v-row justify="end"> -->

          <p>Size</p>
          <v-btn-toggle class="my-1">
            <v-btn text v-on:click="setToolSize(5)">5</v-btn>
            <v-btn text v-on:click="setToolSize(10)">10</v-btn>
            <v-btn text v-on:click="setToolSize(20)">20</v-btn>
          </v-btn-toggle>

          <p>Mode</p>
          <v-btn-toggle class="my-1">
            <v-btn v-on:click="setTool('activate')">
            <v-icon>mdi-eraser</v-icon>
            </v-btn>
            <v-btn v-on:click="setTool('deactivate')">
            <v-icon>mdi-brush</v-icon>
            </v-btn>
          </v-btn-toggle>
          <!-- </v-row> -->
        </v-col>

        <v-col>
          <!--canvas for brain image-->
          <div class="canvas">
            <canvas ref="img" v-bind:width="canvas_width" v-bind:height="canvas_height"></canvas>
          </div>
          <!--canvas for activation map-->
          <div class="canvas">
            <canvas ref="map" v-bind:width="canvas_width" v-bind:height="canvas_height"></canvas>
          </div>
          <div class="canvas">
            <canvas ref="cursor" v-bind:width="canvas_width" v-bind:height="canvas_height"></canvas>
          </div>
          <!--canvas for corrections-->
          <div id="draw">
            <canvas
              ref="draw"
              v-bind:width="canvas_width"
              v-bind:height="canvas_height"
              v-on:mousemove="correctActivation"
            ></canvas>
          </div>
        </v-col>

        <v-col>
          <!-- <div class="thumbnail_image" v-for="image in images" v-bind:key="image.id">
            <img
              :src="require(`../assets/${image.path}.jpeg`)"
              weight="100"
              height="100"
              v-on:click="loadImage(image)"
            />
          </div>-->
        </v-col>
      </v-row>
      <!-- </div> -->

      <v-row alight="center" justify="center">
        <v-sheet light elevation="12" max-width="500" class="ma-8">
          <v-slide-group show-arrows>
            <v-slide-item class="ma-4" v-for="image in images" :key="image.id">
              <v-card width="100">
                <v-img
                  contain
                  :src="require(`../assets/${image.path}.jpeg`)"
                  weight="100"
                  height="100"
                  v-on:click="loadImage(image)"
                />
              </v-card>
            </v-slide-item>
          </v-slide-group>
        </v-sheet>
      </v-row>

      <v-dialog v-model="loading" persistent>
        <v-card color="primary">
          <v-card-text>
            {{ loading_message }}
            <v-progress-linear indeterminate color="white"></v-progress-linear>
          </v-card-text>
        </v-card>
      </v-dialog>
    </v-container>
  </v-app>
</template>>

<script>
import axios from "axios";

export default {
  data() {
    return {
      current_image: {
        image_id: "",
        disease: "",
        path: ""
      },
      image: [],
      images: [
        // {
        //   id: "00001",
        //   disease: "Stroke",
        //   path: "Site6_031923___100"
        // },
      ],
      activationMap: {},
      tool: "deactivate",
      tool_started: false,
      tool_size: 10,
      canvas_width: 436,
      canvas_height: 364,
      graphics: {
        // rgba
        activation_color: [255, 0, 0, 100],
        inactivation_color: [0, 0, 0, 0],
        paintbrush: "rgba(0, 0, 255, 255)",
        image_opacity: 1.0,
        map_opacity: 0.8
      },
      loading: false,
      check_number: 1,
      current_epoch: null,
      loading_message: "Initializing Training..."
    };
  },
  watch: {
    check_number: function() {
      if (this.loading) {
        setTimeout(this.updateTrainingProgress, 10000)
      } else {
        return null
      }
    }
  },
  methods: {
    setTool: function(tool) {
      this.tool = tool;
    },
    setToolSize: function(tool_size) {
      this.tool_size = tool_size;
    },
    getActivationMap: function() {
      const path = "http://localhost:5000/active_learning";
      axios
        .get(path)
        .then(res => {
          this.images = res.data.images;
          this.activationMap = res.data.activation_map;
          // load first image in images
          this.loadImage(this.images[0]);
        })
        .catch(error => {
          console.error(error);
        });
    },
    loadImage: function(image) {
      this.current_image = image;
      let img_ctx = this.$refs.img.getContext("2d");
      // set global opacity
      img_ctx.globalAlpha = this.graphics.image_opacity;
      let img = new Image();
      img.src = require(`../assets/${image.path}.jpeg`);
      img.onload = () => {
        img_ctx.drawImage(img, 0, 0, this.canvas_width, this.canvas_height);
      };
      // find corresponding image id
    },
    loadCursor: function(x, y) {
      let ctx = this.$refs.cursor.getContext("2d");
      ctx.clearRect(0, 0, this.canvas_width, this.canvas_height);
      ctx.fillStyle = "rgba(255,255,255,255)";
      ctx.beginPath();
      ctx.arc(x, y, this.tool_size / 2, 0, 2 * Math.PI);
      ctx.fill();
    },
    loadActivationMap: function() {
      let map_ctx = this.$refs.map.getContext("2d");
      map_ctx.globalAlpha = this.graphics.map_opacity;
      let map_data = map_ctx.createImageData(
        this.canvas_width,
        this.canvas_height
      );
      for (let i = 0; i < map_data.height; i++) {
        for (let k = 0; k < map_data.width; k++) {
          let map_index = 4 * (map_data.width * i + k);
          if (this.activationMap.activation[i][k] === 1) {
            map_data.data[map_index] = this.graphics.activation_color[0];
            map_data.data[map_index + 1] = this.graphics.activation_color[1];
            map_data.data[map_index + 2] = this.graphics.activation_color[2];
            map_data.data[map_index + 3] = this.graphics.activation_color[3];
          } else {
            map_data.data[map_index] = this.graphics.inactivation_color[0];
            map_data.data[map_index + 1] = this.graphics.inactivation_color[1];
            map_data.data[map_index + 2] = this.graphics.inactivation_color[2];
            map_data.data[map_index + 3] = this.graphics.inactivation_color[3];
          }
        }
      }
      map_ctx.putImageData(map_data, 0, 0);
    },
    correctActivation: function(event) {
      // load cursor
      this.loadCursor(event.offsetX, event.offsetY);

      // highlight incorrect activation
      let canvas = this.$refs.draw;
      let ctx = this.$refs.draw.getContext("2d");
      ctx.strokeStyle = this.graphics.paintbrush;
      ctx.fillStyle = this.graphics.paintbrush;
      ctx.globalCompositeOperation = "source-over";
      ctx.lineJoin = "round";
      ctx.lineWidth = this.tool_size;
      // start drawing
      canvas.onmousedown = () => {
        // draw circle
        ctx.beginPath();
        ctx.arc(
          event.offsetX,
          event.offsetY,
          this.tool_size / 2,
          0,
          2 * Math.PI
        );
        ctx.fill();
        this.updateActivationMap();
        this.tool_started = true;
        ctx.beginPath();
        ctx.moveTo(event.offsetX, event.offsetY);
      };
      if (this.tool_started) {
        ctx.lineTo(event.offsetX, event.offsetY);
        ctx.stroke();
        this.updateActivationMap();
      }
      canvas.onmouseup = () => {
        // stop drawing
        if (this.tool_started) {
          this.tool_started = false;
        }
        // draw circle
        ctx.beginPath();
        ctx.arc(
          event.offsetX,
          event.offsetY,
          this.tool_size / 2,
          0,
          2 * Math.PI
        );
        ctx.fill();
        this.updateActivationMap();
      };
    },
    updateActivationMap: function() {
      let map_ctx = this.$refs.map.getContext("2d");
      let draw_ctx = this.$refs.draw.getContext("2d");
      let map_data = map_ctx.getImageData(
        0,
        0,
        this.canvas_width,
        this.canvas_height
      );
      let correction_data = draw_ctx.getImageData(
        0,
        0,
        this.canvas_width,
        this.canvas_height
      );

      let pixel_color = this.graphics.activation_color;
      if (this.tool === "deactivate") {
        pixel_color = this.graphics.inactivation_color;
      }

      let updated_map_data = map_ctx.createImageData(map_data);
      for (let i = 0; i < correction_data.data.length; i += 4) {
        if (correction_data.data[i + 2] === 255) {
          map_data.data[i] = pixel_color[0];
          map_data.data[i + 1] = pixel_color[1];
          map_data.data[i + 2] = pixel_color[2];
          map_data.data[i + 3] = pixel_color[3];
        }
      }

      // draw updated activation map
      map_ctx.putImageData(map_data, 0, 0);

      // clear draw
      let clear_data = map_ctx.createImageData(map_data);
      draw_ctx.putImageData(clear_data, 0, 0);
    },
    saveData: function(payload) {
      const path = "http://localhost:5000/active_learning";
      axios
        .post(path, payload)
        .then(() => {
          //alert("Progress saved!");
          this.loading = false;
          this.getActivationMap();
        })
        .catch(error => {
          console.log(error);
          this.getActivationMap();
        });

      setTimeout(this.updateTrainingProgress, 10000);
    },
    onSubmit: function(event) {
      this.updateActivationMap();
      let map_ctx = this.$refs.map.getContext("2d");
      let corrected_map = map_ctx.getImageData(
        0,
        0,
        this.canvas_width,
        this.canvas_height
      );
      let corrected_activation = [];
      for (let i = 0; i < corrected_map.data.length; i += 4) {
        if (corrected_map.data[i] === 255) {
          corrected_activation.push(1);
        } else {
          corrected_activation.push(0);
        }
      }
      const payload = {
        image: this.current_image,
        corrected_activation: corrected_activation
      };
      this.saveData(payload);
      this.loading = true;
    },
    updateTrainingProgress: function() {
      const path = "http://localhost:5000/training_progress";
      axios
        .get(path)
        .then(res => {
          this.current_epoch = res.data.current_epoch;
          console.log(this.current_epoch);
          this.loading_message =
            "Current epoch:" + String(this.current_epoch) + "/10";
          this.check_number = this.check_number + 1;
        })
        .catch(error => {
          console.error(error);
        });
    }
  },
  created() {
    this.getActivationMap();
  },
  mounted() {}
};
</script>

<style scoped>
#container {
  position: relative;
}
.canvas {
  position: absolute;
  cursor: none;
}
#draw {
  position: relative;
  cursor: none;
  float: left;
}
#graphics {
  position: relative;
  float: left;
  padding: 1em;
}
</style>